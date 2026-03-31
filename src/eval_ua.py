import argparse
import glob
import json
import os
from tqdm import tqdm

# --- Import local modules ---
from metric.ua_metric import StatsCalculator
from ua.evaluate import evaluate
from ua.scoop import SCoOP
from ua.mv import MajorityVoting
from ua.ns import NaiveSelection
# ----------------------------


EPSILON = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse multiple UQ logs using uncertainty aggregation (UA) methods."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to a directory that contains multiple log*.json files.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="scoop",
        choices=["scoop", "ns", "mv"],
        help="Aggregation method: scoop, ns, mv.",
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=2000,
        help="Maximum number of common valid samples to fuse.",
    )
    return parser.parse_args()


def _get_valid_samples(log_path):
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        valid_samples = set()
        for key, value in data.items():
            if not (key.isdigit() and isinstance(value, dict)):
                continue

            benchmark_data = value.get("benchmark", {})
            uq_data = value.get("uq", {})
            if (
                benchmark_data.get("flag_sample_valid") is True
                and "cluster_dict" in uq_data
                and uq_data.get("cluster_dict")
            ):
                valid_samples.add(int(key))

        return valid_samples
    except Exception as exc:
        print(f"Error reading {log_path}: {exc}")
        return set()


def find_common_valid_samples(log_paths, max_samples):
    all_valid_sets = [_get_valid_samples(p) for p in log_paths]
    if not all_valid_sets:
        return []

    common_samples = sorted(list(set.intersection(*all_valid_sets)))
    return common_samples[:max_samples]


def extract_single_sample(sample_idx, log_file_path, log_name, sample_result):
    try:
        with open(log_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        sample_key = str(sample_idx)
        if sample_key in data and isinstance(data[sample_key], dict):
            sample_data = data[sample_key]
            benchmark_data = sample_data.get("benchmark", {})
            uq_data = sample_data.get("uq", {})
            acc_data = sample_data.get("acc", {})

            if benchmark_data.get("flag_sample_valid") is True:
                cluster_dict = uq_data.get("cluster_dict", {})
                ans = acc_data.get("ans")
                label = str(benchmark_data.get("label"))
                latency = sample_data.get("latency", {})
                output_tokens = sample_data.get("output_tokens", {})

                sample_result["label"] = label
                sample_result[log_name] = {
                    "ans": ans,
                    "label": label,
                    "raw_cluster_dict": cluster_dict,
                }
                return sample_result, latency, output_tokens

        return sample_result, {}, {}

    except Exception as exc:
        print(f"Error extracting sample {sample_idx} from {log_file_path}: {exc}")
        return sample_result, {}, {}


def _build_aggregator(agg_method):
    if agg_method == "scoop":
        return SCoOP(EPSILON)
    if agg_method == "ns":
        return NaiveSelection(EPSILON)
    if agg_method == "mv":
        return MajorityVoting(EPSILON)
    raise ValueError(f"Unknown aggregation method: {agg_method}")


def process_single_sample(sample_idx, log_paths, agg_method):
    sample_result = {
        "sample_idx": sample_idx,
        "label": None,
        "fusion_results": {},
        "latency": {},
        "output_tokens": {},
        "error": None,
    }

    num_logs = len(log_paths)
    for i in range(num_logs):
        sample_result[f"log{i+1}_data"] = {}

    log_latencies = []
    log_output_tokens_list = []
    for i, log_path in enumerate(log_paths):
        log_name = f"log{i+1}_data"
        sample_result, latency, output_tokens = extract_single_sample(
            sample_idx, log_path, log_name, sample_result
        )
        log_latencies.append(latency)
        log_output_tokens_list.append(output_tokens)

    all_valid = True
    for i in range(num_logs):
        log_name = f"log{i+1}_data"
        if not sample_result[log_name].get("raw_cluster_dict"):
            all_valid = False
            break

    if not all_valid:
        sample_result["error"] = (
            f"Sample {sample_idx} not found or invalid in one or more logs"
        )
        return sample_result

    try:
        aggregator = _build_aggregator(agg_method)
        sample_result, fusion_latency = aggregator.fuse_per_sample(sample_result)
    except Exception as exc:
        sample_result["error"] = str(exc)
        return sample_result

    stats_calculator = StatsCalculator()
    sample_result = stats_calculator.calculate_latency(
        sample_result, fusion_latency, log_latencies
    )
    sample_result = stats_calculator.calculate_output_tokens(
        sample_result, log_output_tokens_list
    )
    return sample_result


def process_multiple_samples(log_paths, max_samples, agg_method):
    fusion_log_dict = {
        "summary": {
            "log_paths": log_paths,
            "epsilon": EPSILON,
            "aggregation_method": agg_method,
        },
        "entropy_stats": {},
        "latency_stats": {},
        "output_tokens_stats": {},
        "sample_result_list": [],
    }

    common_samples = find_common_valid_samples(log_paths, max_samples)
    if not common_samples:
        fusion_log_dict["error"] = "No common valid samples found"
        return fusion_log_dict

    fusion_log_dict["summary"]["total_samples_processed"] = len(common_samples)

    sample_result_list = []
    valid_result_list = []
    for sample_idx in tqdm(common_samples, desc="Processing samples"):
        sample_result = process_single_sample(sample_idx, log_paths, agg_method)
        sample_result_list.append(sample_result)
        if not sample_result.get("error"):
            valid_result_list.append(sample_result)

    fusion_log_dict["sample_result_list"] = sample_result_list

    if valid_result_list:
        stats_calculator = StatsCalculator()
        fusion_log_dict = stats_calculator.calculate_stats(
            fusion_log_dict, valid_result_list
        )

    return fusion_log_dict


def eval_ua(args):
    result_dir = args.result_dir
    max_samples = args.max_sample
    agg_method = args.aggregation

    log_pattern = os.path.join(result_dir, "log*.json")
    log_paths = sorted(glob.glob(log_pattern))
    num_logs = len(log_paths)

    if not (2 <= num_logs <= 10):
        print(
            f"Error: Found {num_logs} log files matching '{log_pattern}'. "
            "This script supports 2 to 10 logs."
        )
        return

    output_dir = os.path.join(result_dir, agg_method)
    output_path = os.path.join(output_dir, "fusion_output.json")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    for i, log_path in enumerate(log_paths):
        print(f"Log {i + 1}: {log_path}")
    print(f"Max samples: {max_samples}")
    print("Start fusing...")

    fusion_log_dict = process_multiple_samples(log_paths, max_samples, agg_method)

    if "error" in fusion_log_dict:
        print(f"Error: {fusion_log_dict['error']}")
    else:
        print(
            f"Successfully processed "
            f"{fusion_log_dict['summary']['total_samples_processed']} samples"
        )

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(fusion_log_dict, file, indent=4, ensure_ascii=False)
        print(f"Full fusion log is saved at: {output_path}")
    except Exception as exc:
        print(f"Error saving fusion log: {exc}")
        return

    evaluate(output_dir)


def main():
    args = parse_args()
    eval_ua(args)


if __name__ == "__main__":
    main()
