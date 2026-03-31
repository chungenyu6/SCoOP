import os
import numpy as np
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc


class StatsCalculator():
    def calculate_latency(self, sample_result, fusion_latency, log_latencies):
        """
        Calculate overall latency statistics for N logs.
        Args:
            sample_result (dict): Result dictionary to populate latency info
            fusion_latency (float): Latency of the fusion process
            log_latencies (List[dict]): List of latency info dicts from all logs
        Returns:
            dict: Updated result dictionary with latency info
        """
        try:
            num_logs = len(log_latencies)
            infer_latencies = [log.get('init_infer_latency', 0) for log in log_latencies]
            e2e_latencies = [log.get('end_to_end_latency', 0) for log in log_latencies]

            overall_latency = sum(infer_latencies) # N LVLMs without UQ without fusion
            overall_latency_uq = sum(e2e_latencies) # N LVLMs with UQ without fusion
            overall_latency_uq_fusion = overall_latency_uq + fusion_latency # N LVLMs with UQ with fusion

            sample_result['latency'] = {
                'fusion_latency': fusion_latency,
                'overall_latency': overall_latency,
                'overall_latency_uq': overall_latency_uq,
                'overall_latency_uq_fusion': overall_latency_uq_fusion
            }
            
            # --- Add individual log latencies dynamically ---
            for i in range(num_logs):
                sample_result['latency'][f'log{i+1}_infer_latency'] = infer_latencies[i]
                sample_result['latency'][f'log{i+1}_end_to_end_latency'] = e2e_latencies[i]
            # ---------------------------------------------
            
        except Exception as e:
            sample_result['latency'] = {
                'error': str(e)
            }
            
        return sample_result

    def calculate_output_tokens(self, sample_result, log_output_tokens_list):
        """
        Calculate output tokens statistics from N logs.
        
        Args:
            sample_result (dict): Result dictionary to populate output tokens info
            log_output_tokens_list (List[dict]): List of output tokens info dicts from all logs
        Returns:
            dict: Updated result dictionary with output tokens info
        """

        try:
            num_logs = len(log_output_tokens_list)
            init_output_tokens = [log.get('init_output_tokens', 0) for log in log_output_tokens_list]
            total_output_tokens = [log.get('total_output_tokens', 0) for log in log_output_tokens_list]

            overall_output_tokens_fusion = sum(init_output_tokens) # N LVLMs without UQ with fusion
            overall_output_tokens_uq_fusion = sum(total_output_tokens) # N LVLMs with UQ with fusion
            
            sample_result['output_tokens'] = {
                'overall_output_tokens_fusion': overall_output_tokens_fusion,
                'overall_output_tokens_uq_fusion': overall_output_tokens_uq_fusion
            }
            
            # --- Add individual log tokens dynamically ---
            for i in range(num_logs):
                sample_result['output_tokens'][f'log{i+1}_init_output_tokens'] = init_output_tokens[i]
                sample_result['output_tokens'][f'log{i+1}_total_output_tokens'] = total_output_tokens[i]
            # ------------------------------------------
            
        except Exception as e:
            sample_result['output_tokens'] = {
                'error': str(e)
            }
        
        return sample_result

    def calculate_stats(self, fusion_log_dict, valid_result_list):
        if not valid_result_list:
            return fusion_log_dict
            
        # --- Robustly determine num_logs from the first valid sample ---
        sample_latency_keys = valid_result_list[0].get('latency', {}).keys()
        num_logs = sum(1 for key in sample_latency_keys if key.startswith('log') and key.endswith('_infer_latency'))
        
        if num_logs == 0:
            print("Warning: Could not determine number of logs from stats in calculate_stats.")
            return fusion_log_dict
        # -------------------------------------------------------------
        
        # --- Helper function to avoid repetition ---
        def get_stats_dict(data_list):
            if not data_list:
                return {'mean': 0, 'median': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'min': 0, 'max': 0}
            return {
                'mean': float(np.mean(data_list)),
                'median': float(np.median(data_list)),
                'p50': float(np.percentile(data_list, 50)),
                'p95': float(np.percentile(data_list, 95)),
                'p99': float(np.percentile(data_list, 99)),
                'min': float(np.min(data_list)),
                'max': float(np.max(data_list))
            }
        # ------------------------------------------

        # ------------------------------
        # 1. Calculate entropy statistics
        all_entropies = []
        for i in range(num_logs):
            all_entropies.append([r['fusion_results'][f'log{i+1}_entropy'] for r in valid_result_list])
        entropies_fused = [r['fusion_results']['fused_entropy'] for r in valid_result_list]
        
        fusion_log_dict['entropy_stats'] = {
            'avg_entropy_fused': np.mean(entropies_fused),
            'std_entropy_fused': np.std(entropies_fused),
        }
        for i in range(num_logs):
            fusion_log_dict['entropy_stats'][f'avg_entropy_log{i+1}'] = np.mean(all_entropies[i])
            fusion_log_dict['entropy_stats'][f'std_entropy_log{i+1}'] = np.std(all_entropies[i])
        # ------------------------------
        
        # 2. Calculate latency statistics
        all_infer_latencies = []
        for i in range(num_logs):
            all_infer_latencies.append([r['latency'][f'log{i+1}_infer_latency'] for r in valid_result_list])
            
        overall_latencies = [r['latency']['overall_latency'] for r in valid_result_list]
        overall_latencies_uq = [r['latency']['overall_latency_uq'] for r in valid_result_list]
        overall_latencies_with_fusion = [r['latency']['overall_latency_uq_fusion'] for r in valid_result_list]
        
        fusion_log_dict['latency_stats'] = {}
        for i in range(num_logs):
            fusion_log_dict['latency_stats'][f'log{i+1}_infer_latency_stats'] = get_stats_dict(all_infer_latencies[i])
        
        fusion_log_dict['latency_stats']['overall_latency_stats'] = get_stats_dict(overall_latencies)
        fusion_log_dict['latency_stats']['overall_latency_uq_stats'] = get_stats_dict(overall_latencies_uq)
        fusion_log_dict['latency_stats']['overall_latency_uq_fusion_stats'] = get_stats_dict(overall_latencies_with_fusion)
        # ------------------------------
        
        # 3. Calculate output tokens statistics
        all_init_output_tokens = []
        for i in range(num_logs):
            all_init_output_tokens.append([r['output_tokens'][f'log{i+1}_init_output_tokens'] for r in valid_result_list])

        overall_output_tokens_fusion_list = [r['output_tokens']['overall_output_tokens_fusion'] for r in valid_result_list]
        overall_output_tokens_uq_fusion_list = [r['output_tokens']['overall_output_tokens_uq_fusion'] for r in valid_result_list]
        
        fusion_log_dict['output_tokens_stats'] = {}
        for i in range(num_logs):
            fusion_log_dict['output_tokens_stats'][f'log{i+1}_init_output_tokens_stats'] = get_stats_dict(all_init_output_tokens[i])

        fusion_log_dict['output_tokens_stats']['overall_output_tokens_fusion_stats'] = get_stats_dict(overall_output_tokens_fusion_list)
        fusion_log_dict['output_tokens_stats']['overall_output_tokens_uq_fusion_stats'] = get_stats_dict(overall_output_tokens_uq_fusion_list)
        # ------------------------------

        return fusion_log_dict


class MetricCalculator():
    def compute_fusion_accuracy(self, per_sample: List[Dict[str, Any]]) -> float:
        """(No changes)"""
        correct = sum(1 for s in per_sample if s["fusion_correct"])
        total = len(per_sample)
        return correct / total if total > 0 else float("nan")

    def compute_fusion_auroc(self, per_sample: List[Dict[str, Any]]) -> float | None:
        """
        AUROC with:
        y_true = 1 if fused answer incorrect; else 0.
        y_score = fused_entropy (higher => more likely incorrect).
        Returns None if all answers share the same correctness label.
        (No changes)
        """
        y_true = [0 if s["fusion_correct"] else 1 for s in per_sample]
        if len(set(y_true)) < 2:
            return None
        y_score = [s["fused_entropy"] for s in per_sample]
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            return None # Handle edge cases where roc_auc_score fails

    def compute_fusion_aurac(self, uncertainties, correctness, num_points):
        """Compute Area Under the Rejection-Accuracy Curve. (No changes)"""
        uncertainties = np.array(uncertainties)
        correctness = np.array(correctness)
        n = len(correctness)
        if n == 0:
            return 0.0

        sorted_idx = np.argsort(-uncertainties) # Sort by uncertainty descending
        correctness_sorted = correctness[sorted_idx]
        
        rejection_rates = np.linspace(0, 1, num_points)
        accuracies = []
        for r in rejection_rates:
            k = int(r * n)
            # Handle edge case where r=1.0, k=n
            if k == n:
                kept = []
            else:
                kept = correctness_sorted[k:] # Keep items with *lower* uncertainty
            
            acc = np.mean(kept) if len(kept) > 0 else 0.0
            accuracies.append(acc)
        return float(np.trapz(accuracies, rejection_rates))