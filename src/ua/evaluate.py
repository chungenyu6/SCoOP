"""
Compute fusion metrics from fusion_output.json:
1) Fusion Accuracy
2) Fusion AUROC
4) Fusion AURAC

Outputs fusion_result.json with metrics and per-sample breakdown.
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import os
from typing import Any, Dict, List, Tuple

# Allow running this file directly via: python src/ua/evaluate.py ...
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --- Import local modules ---
from metric.ua_metric import MetricCalculator
# ----------------------------

NUM_POINTS = 50


def load_fusion_output(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "sample_result_list" not in doc or not isinstance(doc["sample_result_list"], list):
        raise ValueError("Invalid input: missing or non-list 'sample_result_list'.")
    if "entropy_stats" not in doc or not isinstance(doc["entropy_stats"], dict):
        raise ValueError("Invalid input: missing 'entropy_stats'.")
    if "avg_entropy_fused" not in doc["entropy_stats"]:
        raise ValueError("Invalid input: missing 'entropy_stats.avg_entropy_fused'.")
    return doc


def _is_finite_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def extract_sample_fields(sample: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any], float, Dict[str, Any]]:
    if sample.get("error") is not None:
        raise ValueError(f"Sample has error: {sample.get('error')}")

    idx = int(sample.get("sample_idx", -1))
    label = str(sample["label"])
    fusion_result_dict = sample.get("fusion_results", {})
    if not isinstance(fusion_result_dict, dict):
        raise ValueError("Missing 'fusion_results'.")

    fused_probability_vector = fusion_result_dict.get("fused_probability_vector")
    if not isinstance(fused_probability_vector, dict) or not fused_probability_vector:
        raise ValueError("Missing or empty 'fused_probability_vector'.")

    fused_entropy = fusion_result_dict.get("fused_entropy")
    if not _is_finite_number(fused_entropy):
        raise ValueError("Missing or non-finite 'fused_entropy'.")

    return idx, label, fused_probability_vector, float(fused_entropy), fusion_result_dict


def evaluate(result_dir: str) -> None:
    input_path = os.path.join(result_dir, "fusion_output.json")
    output_path = os.path.join(result_dir, "fusion_result.json")

    doc = load_fusion_output(input_path)
    raw_samples: List[Dict[str, Any]] = doc["sample_result_list"]

    per_sample: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    correctness = []
    uncertainties = []

    for sample in raw_samples:
        try:
            idx, label, fpv, fused_entropy, fusion_result_dict = extract_sample_fields(sample)
            fused_ans = fusion_result_dict.get("fused_ans")
            fusion_correct = fused_ans == label

            correctness.append(1 if fusion_correct else 0)
            uncertainties.append(fused_entropy)

            per_sample.append(
                {
                    "sample_idx": idx,
                    "label": label,
                    "fused_ans": fused_ans,
                    "fusion_correct": fusion_correct,
                    "fused_entropy": fused_entropy,
                }
            )
        except Exception as exc:
            skipped.append(
                {
                    "sample_idx": sample.get("sample_idx", None),
                    "reason": str(exc),
                }
            )

    if not per_sample:
        print(f"No valid samples found in {input_path}. Skipping metrics.")
        return

    metric_calculator = MetricCalculator()
    fusion_acc = metric_calculator.compute_fusion_accuracy(per_sample) * 100
    fusion_auroc = metric_calculator.compute_fusion_auroc(per_sample)
    fusion_aurac = metric_calculator.compute_fusion_aurac(uncertainties, correctness, NUM_POINTS)

    result = {
        "source_summary": {
            "total_samples_in_file": len(raw_samples),
            "total_used": len(per_sample),
            "total_skipped": len(skipped),
            "log_paths": doc.get("summary", {}).get("log_paths", []),
            "epsilon": doc.get("summary", {}).get("epsilon"),
        },
        "metrics": {
            "fusion_accuracy": fusion_acc,
            "fusion_auroc": fusion_auroc,
            "fusion_aurac": fusion_aurac,
        },
        "per_sample": per_sample,
        "skipped_samples": skipped,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("-" * 30)
    print("Fusion Performance Summary")
    print("-" * 30)
    print(f"Total Samples   : {result['source_summary']['total_samples_in_file']}")
    print(f"Fusion Accuracy : {result['metrics']['fusion_accuracy']:.6f}")
    print(
        f"Fusion AUROC    : "
        f"{result['metrics']['fusion_auroc'] if result['metrics']['fusion_auroc'] is not None else 'N/A'}"
    )
    print(f"Fusion AURAC    : {result['metrics']['fusion_aurac']:.6f}")
    print(f"Save results to : {os.path.abspath(output_path)}")
    print("=" * 60)
    print("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fusion output metrics.")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing fusion_output.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate(cli_args.result_dir)
