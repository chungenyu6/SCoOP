"""
# l2-categories (broad) and categories

## finegrained_perception (cross-instance)
- action_recognition - 215
- attribute_comparison - 141
- spatial_relationship - 177

## finegrained_perception (instance-level)
- attribute_recognition - 264
- celebrity_recognition - 396
- object_localization - 315
- ocr - 156

## attribute_reasoning
- function_reasoning - 304
- identity_reasoning - 176
- physical_property_reasoning - 219

## logic_reasoning
- future_prediction - 139
- structured_imagetext_understanding - 282

## coarse_perception
- image_emotion - 200
- image_quality - 150
- image_scene - 487
- image_style - 212
- image_topic - 140

## relation_reasoning
- nature_relation - 179
- physical_relation - 94
- social_relation - 172
"""

import sys
import os
import base64
import io

try:
    # Requires 'datasets' library: pip install datasets
    from datasets import load_dataset
except ImportError:
    print("Hugging Face 'datasets' library not found. Please install it with 'pip install datasets'", file=sys.stderr)
    sys.exit(1)

try:
    # Requires 'Pillow' library: pip install Pillow
    from PIL import Image
except ImportError:
    print("PIL (Pillow) library not found. Please install it with 'pip install Pillow'", file=sys.stderr)
    sys.exit(1)


class MMBench:
    """
    A benchmark class for MMBench, structured similarly to the MMMU class.
    
    This class loads data from the HuggingFaceM4/MMBench dataset on the Hub,
    filters for specified l2-categories, and formats samples as needed.
    """

    def __init__(self):
        """
        Initializes the MMBench benchmark.
        """
        
        # --- 1. Configuration ---
        # TEST
        # self.categories = [
        #     "attribute_recognition",
        #     "celebrity_recognition",
        #     "object_localization",
        #     "ocr",
        # ]
        # NOTE: official
        self.categories = [
            ## finegrained_perception (cross-instance)
            "action_recognition",
            "attribute_comparison",
            "spatial_relationship",

            ## finegrained_perception (instance-level)
            "attribute_recognition",
            "celebrity_recognition",
            "object_localization",
            "ocr",

            ## attribute_reasoning
            "function_reasoning",
            "identity_reasoning",
            "physical_property_reasoning",

            ## logic_reasoning
            "future_prediction",
            "structured_imagetext_understanding",

            ## coarse_perception
            "image_emotion",
            "image_quality",
            "image_scene",
            "image_style",
            "image_topic",

            ## relation_reasoning
            "nature_relation",
            "physical_relation",
            "social_relation",
        ]
        self.NUM_SAMPLES_EACH_CATEGORY = 50 
        self.NUM_CATEGORY = len(self.categories)
        
        # Map string labels ('A', 'B', ...) to integer indices (0, 1, ...)
        self.label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.choice_indices = ["0", "1", "2", "3"] # MMBench always has A, B, C, D
        self.choice_numbers_str = ", ".join(self.choice_indices)

        # --- 2. Load Data from Hugging Face Hub ---
        print("Loading MMBench dataset from Hugging Face Hub...")
        try:
            # Load the entire validation split
            ds = load_dataset('HuggingFaceM4/MMBench', split='validation')
        except Exception as e:
            print(f"Error loading dataset 'HuggingFaceM4/MMBench': {e}", file=sys.stderr)
            print("Please check your internet connection and if the dataset name is correct.", file=sys.stderr)
            sys.exit(1)
        print("Dataset loaded successfully.")

        # --- 3. Pre-process and Filter Samples ---
        self.samples_to_process = []
        
        # Group samples by category for efficient sampling
        category_map = {cat: [] for cat in self.categories}
        
        print("Filtering and grouping samples...")
        for row in ds:
            # Change 'l2-category' to 'category'
            category = row['category'] 
            # Check if this row is from a category we want
            if category in category_map:
                # Check if we still need samples for this category
                if len(category_map[category]) < self.NUM_SAMPLES_EACH_CATEGORY:
                    category_map[category].append(row)
        
        # Flatten the map into the final processing list
        for category in self.categories:
            cat_samples = category_map[category]
            self.samples_to_process.extend(cat_samples)
            if len(cat_samples) < self.NUM_SAMPLES_EACH_CATEGORY:
                print(f"Warning: Found only {len(cat_samples)} samples for category '{category}' (requested {self.NUM_SAMPLES_EACH_CATEGORY}).", file=sys.stderr)
        
        print(f"Initialization complete. Total samples to process: {len(self.samples_to_process)}")


    def _assemble_question(self, row):
        """Formats the question and options into the MMMU prompt style."""
        
        question = row['question']
        hint = row['hint']
        options = [row['A'], row['B'], row['C'], row['D']]
        
        choices_str = ""
        for i, opt in enumerate(options):
            choices_str += f'({i}): {opt}\n'
            
        # Add hint if it exists and is not empty
        hint_str = f"HINT: {hint}\n" if hint else ""

        # Assemble the final question in MMMU format
        question += '\n'
        question += hint_str
        question += '\n'
        question += choices_str
        question += '\n'
        question += f'This is a single choice question, answer only with choice number in {self.choice_numbers_str}.'
        # msg = f"""{question}
        # {choices_str}
        # This is a single choice question, answer only with choice number in {self.choice_numbers_str}."""
        
        # return msg
        return question

    def obtain_size(self):
        """Returns the total number of samples to be evaluated."""
        return len(self.samples_to_process)

    def retrieve(self, idx):
        """
        Retrieves and formats a single sample by its sequential index.
        
        Args:
            idx (int): The index in the range [0, obtain_size() - 1].
            
        Returns:
            A dictionary containing the formatted sample, or None if retrieval fails.
        """
        if idx >= len(self.samples_to_process):
            return None

        row = self.samples_to_process[idx]

        # --- Load Image ---
        # The 'image' field is a Base64 encoded string.
        base64_string = row['image']
        
        # Check if it's a valid string
        if not base64_string or not isinstance(base64_string, str):
            print(f"Warning: Sample at index {idx} (original index {row['index']}) has no valid image string. Skipping.", file=sys.stderr)
            return None

        try:
            # Decode the Base64 string into binary data
            binary_data = base64.b64decode(base64_string)
            # Create an in-memory file from the binary data
            image_file = io.BytesIO(binary_data)
            # Open the image using PIL
            img = Image.open(image_file)
        except Exception as e:
            print(f"Error decoding Base64 image for original index {row['index']}: {e}. Skipping.", file=sys.stderr)
            return None
            
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # --- Format Question ---
        question_formatted = self._assemble_question(row)

        # --- Format Label ---
        label_str = row['answer']
        label = self.label_mapping.get(label_str)
        if label is None:
            print(f"Warning: Unrecognized label '{label_str}' for original index {row['index']}. Skipping.", file=sys.stderr)
            return None

        # --- Construct final sample dict ---
        result = {
            'idx': idx, # Use the sequential index for the evaluation loop
            'img': img,
            'question': question_formatted,
            'label': label,
            'num_c': 4, # MMBench always has 4 (A, B, C, D)
            'original_index': row['index'] # Keep original MMBench index for logging
        }
        
        return result

# --- Example Usage  ---
if __name__ == "__main__":
    from lvlm.lvlm_router import LVLMRouter

    lvlm = LVLMRouter(backend="gemma3", version="gemma-3-4b-it", gpu_mem_util=0.5)

    print(f"--- Initializing MMBench Benchmark ---")
    # 1. Initialize the benchmark
    #    (To test, uncomment the TEST variables in __init__ for a quick run)
    benchmark = MMBench()

    # 2. Test obtain_size()
    total_size = benchmark.obtain_size()
    print(f"\nBenchmark total size (from {benchmark.NUM_CATEGORY} categories): {total_size}")

    # 3. Test retrieve(0)
    if total_size > 0:
        print("\n--- Retrieving sample 0 ---")
        first_sample = benchmark.retrieve(0)
        
        if first_sample:
            print(f"Index (sequential): {first_sample['idx']}")
            print(f"Index (original MMBench): {first_sample['original_index']}")
            print(f"Image object: {first_sample['img']}")
            print(f"Label (numeric): {first_sample['label']}")
            print(f"Num choices: {first_sample['num_c']}")
            print("\nQuestion:\n")
            print(first_sample['question'])

            ans, ans_neg_logprob, infer_latency, output_tokens = lvlm.generate(
                first_sample['img'],
                # first_sample['question'],
                "caption the image in details",
                0.1
            )
            print("model sresponse: \n", ans)
        else:
            print("Failed to retrieve sample 0.")
    else:
        print("No samples loaded, cannot retrieve sample 0.")
