import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

# --- Import local modules ---
from benchmark.MMBench import MMBench
from benchmark.MMMU import MMMU
from benchmark.ScienceQA import ScienceQA
from lvlm.lvlm_router import LVLMRouter
from metric.uq_metric import MetricCalculator
from uq.estimate import route_uq_method
# ----------------------------

warnings.filterwarnings("ignore")


LVLM_MAP = {
    "llava-v1.6-mistral-7b-hf": lambda: LVLMRouter(backend="llava1_6", version="llava-v1.6-mistral-7b-hf", gpu_mem_util=0.5),
    "llava-v1.6-vicuna-13b-hf": lambda: LVLMRouter(backend="llava1_6", version="llava-v1.6-vicuna-13b-hf", gpu_mem_util=0.5),
    "llava-v1.6-34b-hf": lambda: LVLMRouter(backend="llava1_6", version="llava-v1.6-34b-hf", gpu_mem_util=0.7),
    "llava-next-72b-hf": lambda: LVLMRouter(backend="llava1_6", version="llava-next-72b-hf", gpu_mem_util=0.9),
    "gemma-3-4b-it": lambda: LVLMRouter(backend="gemma3", version="gemma-3-4b-it", gpu_mem_util=0.5),
    "gemma-3-12b-it": lambda: LVLMRouter(backend="gemma3", version="gemma-3-12b-it", gpu_mem_util=0.5),
    "gemma-3-27b-it": lambda: LVLMRouter(backend="gemma3", version="gemma-3-27b-it", gpu_mem_util=0.5),
    "InternVL3-2B-Instruct": lambda: LVLMRouter(backend="internvl3", version="InternVL3-2B-Instruct", gpu_mem_util=0.5),
    "InternVL3-8B-Instruct": lambda: LVLMRouter(backend="internvl3", version="InternVL3-8B-Instruct", gpu_mem_util=0.5),
    "InternVL3-14B-Instruct": lambda: LVLMRouter(backend="internvl3", version="InternVL3-14B-Instruct", gpu_mem_util=0.5),
    "InternVL3-38B-Instruct": lambda: LVLMRouter(backend="internvl3", version="InternVL3-38B-Instruct", gpu_mem_util=0.5),
    "InternVL3-78B-Instruct": lambda: LVLMRouter(backend="internvl3", version="InternVL3-78B-Instruct", gpu_mem_util=0.9),
    "deepseek-vl2-tiny": lambda: LVLMRouter(backend="deepseekvl2", version="deepseek-vl2-tiny", gpu_mem_util=0.5),
    "deepseek-vl2-small": lambda: LVLMRouter(backend="deepseekvl2", version="deepseek-vl2-small", gpu_mem_util=0.5),
    "deepseek-vl2": lambda: LVLMRouter(backend="deepseekvl2", version="deepseek-vl2", gpu_mem_util=0.4),
    "Qwen2.5-VL-3B-Instruct": lambda: LVLMRouter(backend="qwen2_5", version="Qwen2.5-VL-3B-Instruct", gpu_mem_util=0.5),
    "Qwen2.5-VL-7B-Instruct": lambda: LVLMRouter(backend="qwen2_5", version="Qwen2.5-VL-7B-Instruct", gpu_mem_util=0.5),
    "Qwen2.5-VL-32B-Instruct": lambda: LVLMRouter(backend="qwen2_5", version="Qwen2.5-VL-32B-Instruct", gpu_mem_util=0.7),
    "Qwen2.5-VL-72B-Instruct": lambda: LVLMRouter(backend="qwen2_5", version="Qwen2.5-VL-72B-Instruct", gpu_mem_util=0.9),
}

BENCHMARK_MAP = {
    "MMMU": MMMU,
    "ScienceQA": ScienceQA,
    "MMBench": MMBench,
}

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="ScienceQA")
    parser.add_argument("--lvlm", type=str, default="InternVL3-2B-Instruct")
    parser.add_argument("--uq_method", type=str, default="semantic_entropy")
    parser.add_argument("--inference_temp", type=float, default=0.1)
    parser.add_argument("--sampling_temp", type=float, default=0.1)
    parser.add_argument("--sampling_time", type=int, default=5)
    parser.add_argument(
        "--max_sample",
        type=int,
        default=None,
        help="Optional cap on the number of benchmark samples to process.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="result/uq",
        help="Root directory to save results.",
    )
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def obtain_benchmark(args):
    print(f"Loading Benchmark: {args.benchmark}...")
    benchmark_class = BENCHMARK_MAP.get(args.benchmark)
    if not benchmark_class:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    return benchmark_class()


def obtain_lvlm(args):
    print(f"Loading LVLM: {args.lvlm}...")
    lvlm_builder = LVLM_MAP.get(args.lvlm)
    if not lvlm_builder:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")
    return lvlm_builder()


def obtain_single_sample(args, benchmark, idx, log_dict):
    sample = benchmark.retrieve(idx)
    if sample is None:
        return None

    log_dict[idx]["benchmark"]["question"] = sample["question"]
    log_dict[idx]["benchmark"]["label"] = sample["label"]
    return sample


def infer_lvlm_ans(args, lvlm, sample, log_dict):
    ans, ans_neg_logprob, infer_latency, output_tokens = lvlm.generate(
        sample["img"],
        sample["question"],
        args.inference_temp,
    )

    log_dict[sample["idx"]]["acc"]["ans"] = ans
    log_dict[sample["idx"]]["uq"]["ans_neg_logprob"] = ans_neg_logprob
    log_dict[sample["idx"]]["latency"]["init_infer_latency"] = infer_latency
    log_dict[sample["idx"]]["output_tokens"]["init_output_tokens"] = output_tokens


def process_single_sample(args, idx, lvlm, benchmark, log_dict):
    sample = obtain_single_sample(args, benchmark, idx, log_dict)

    if sample is None or sample["img"] is None or sample["question"] is None or sample["label"] is None:
        log_dict[idx]["benchmark"]["flag_sample_valid"] = False
        return
    log_dict[idx]["benchmark"]["flag_sample_valid"] = True

    infer_lvlm_ans(args, lvlm, sample, log_dict)
    route_uq_method(args, lvlm, sample, log_dict)


def eval_uq(args, lvlm, benchmark):
    benchmark_size = benchmark.obtain_size()
    if args.max_sample is not None:
        benchmark_size = min(benchmark_size, args.max_sample)

    log_dict = {
        "args": vars(args),
        "result": {},
    }

    metric_calculator = MetricCalculator(args)
    start_time = time.time()

    for idx in tqdm(range(benchmark_size)):
        log_dict[idx] = {
            "benchmark": {},
            "acc": {},
            "uq": {},
            "latency": {},
            "output_tokens": {},
        }

        process_single_sample(args, idx, lvlm, benchmark, log_dict)

        if log_dict[idx]["benchmark"]["flag_sample_valid"]:
            metric_calculator.calculate_single(args, idx, log_dict)

        torch.cuda.empty_cache()

    result = metric_calculator.calculate_all()
    total_sample = result["total_sample"]

    end_time = time.time()
    total_runtime_minute = (end_time - start_time) / 60
    log_dict["result"] = {"total_runtime_minute": total_runtime_minute, **result}

    result_dir = os.path.join(
        args.result_root,
        args.benchmark,
        f"total{total_sample}",
        f"itemp{args.inference_temp}",
        f"stemp{args.sampling_temp}",
        f"stime{args.sampling_time}",
        args.uq_method,
        args.lvlm,
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    log_file = os.path.join(result_dir, "log.json")
    result_file = os.path.join(result_dir, "result.json")

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=4)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(log_dict["result"], f, indent=4)

    print(f"Full log is saved at {log_file}.")
    print(f"Full result is saved at {result_file}.")


def main():
    args = parse_args()
    fix_seed(args.seed)

    benchmark = obtain_benchmark(args)
    lvlm = obtain_lvlm(args)

    eval_uq(args, lvlm, benchmark)


if __name__ == "__main__":
    main()
