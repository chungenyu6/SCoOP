"""
Model collections:
    - LLaVA-v1.6: https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf
    - Gemma-3: https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
    - InternVL3: https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d
    - DeepSeek-VL2: https://huggingface.co/collections/deepseek-ai/deepseek-vl2-675c22accc456d3beb4613ab
    - Qwen2.5-VL: https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5
"""

import time
from typing import Literal

import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.multimodal.image import convert_image_mode


class LVLMRouter:
    def __init__(
        self,
        backend: Literal["llava1_6", "gemma3", "internvl3", "deepseekvl2", "qwen2_5"],
        version: str = None,
        modality: Literal["image", "video"] = "image",
        gpu_mem_util: float = 0.9,
        generation_max_tokens: int = 128,
        **engine_kwargs,
    ):
        """
        Unified wrapper for running different multimodal VL models with vLLM.

        Args:
            backend: Model family.
            version: Specific checkpoint name.
            modality: "image" or "video".
            gpu_mem_util: Fraction of GPU memory to allocate.
            generation_max_tokens: Max generated tokens per response.
            engine_kwargs: Extra vLLM engine args.
        """
        torch.cuda.empty_cache()

        self.backend = backend
        self.version = version
        self.modality = modality
        self.gpu_mem_util = gpu_mem_util

        if self.backend == "llava1_6":
            model_name = f"llava-hf/{version}"
            default_kwargs = {
                "max_model_len": 4096,
            }
            vicuna_chat_template = """\
            {% for message in messages %}
            {% if message['role'] == 'user' %}
            USER: {{ message['content'] }}
            {% elif message['role'] == 'assistant' %}
            ASSISTANT: {{ message['content'] }}
            {% endif %}
            {% endfor %}
            ASSISTANT:"""
            self.llava_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.llava_tokenizer.chat_template = vicuna_chat_template

        elif self.backend == "gemma3":
            model_name = f"google/{version}"
            default_kwargs = {
                "mm_processor_kwargs": {"do_pan_and_scan": True},
                "limit_mm_per_prompt": {"image": 1},
                "max_model_len": 2048,
                "max_num_seqs": 2,
            }

        elif self.backend == "internvl3":
            model_name = f"OpenGVLab/{version}"
            default_kwargs = {
                "trust_remote_code": True,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
                "max_model_len": 8192,
            }

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
            stop_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]
            self.stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

        elif self.backend == "deepseekvl2":
            model_name = f"deepseek-ai/{version}"
            default_kwargs = {
                "hf_overrides": {"architectures": ["DeepseekVLV2ForCausalLM"]},
                "limit_mm_per_prompt": {"image": 1},
                "max_model_len": 4096,
                "max_num_seqs": 2,
            }

        elif self.backend == "qwen2_5":
            model_name = f"Qwen/{version}"
            default_kwargs = {
                "mm_processor_kwargs": {
                    "min_pixels": 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                    "fps": 1,
                },
                "limit_mm_per_prompt": {self.modality: 1},
                "max_model_len": 4096,
                "max_num_seqs": 5,
            }
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.lvlm = LLM(
            model=model_name,
            gpu_memory_utilization=self.gpu_mem_util,
            **default_kwargs,
            **engine_kwargs,
        )

        self.sampling_params = SamplingParams(
            max_tokens=generation_max_tokens,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            stop_token_ids=getattr(self, "stop_token_ids", None),
            logprobs=1,
        )

    def load_image(self, image_input):
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        return convert_image_mode(pil_img, "RGB")

    def build_prompt(self, question: str) -> str:
        if self.backend == "llava1_6":
            messages = [{"role": "user", "content": "<image>\n" + question}]
            return self.llava_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if self.backend == "gemma3":
            return (
                "<bos><start_of_turn>user\n"
                f"<start_of_image>{question}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

        if self.backend == "internvl3":
            placeholder = "<image>" if self.modality == "image" else "<video>"
            messages = [{"role": "user", "content": f"{placeholder}\n{question}"}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if self.backend == "deepseekvl2":
            return f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"

        if self.backend == "qwen2_5":
            placeholder = "<|image_pad|>" if self.modality == "image" else "<|video_pad|>"
            return (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        raise ValueError(f"Unsupported backend: {self.backend}")

    def generate(self, image_input, question: str, temperature=None):
        start_time = time.time()

        if temperature is not None:
            self.sampling_params.temperature = temperature

        image = self.load_image(image_input)
        prompt = self.build_prompt(question)

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
            "multi_modal_uuids": {"image": "img_0"},
        }

        outputs = self.lvlm.generate(
            [inputs],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        response_text = outputs[0].outputs[0].text

        seq_logprob = outputs[0].outputs[0].cumulative_logprob
        if seq_logprob is not None:
            logprobs = outputs[0].outputs[0].logprobs
            sequence_length = len(logprobs)
            normalized_neg_logprob = -seq_logprob / sequence_length if sequence_length > 0 else 0.0
        else:
            normalized_neg_logprob = 0.0

        tokenizer = self.lvlm.get_tokenizer()
        output_tokens = len(tokenizer.encode(response_text))
        infer_latency = time.time() - start_time

        return response_text, normalized_neg_logprob, infer_latency, output_tokens
