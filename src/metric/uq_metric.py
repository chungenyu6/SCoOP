import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


BENCHMARK_TYPE = {
    'MMMU': 'MULTI_CHOICE',
    'ScienceQA': 'MULTI_CHOICE',
    'MMBench': 'MULTI_CHOICE',
}


class MetricCalculator:
    def __init__(self, args):
        self.args = args
        self.total_sample = 0
        self.cnt_correct_ans = 0 # for benchmark accuracy
        self.uncertainty_scores = [] # for uq (auroc, aurac)
        self.auroc_labels = [] # for auroc
        self.correctness_labels = [] # for aurac
        self.num_points = 100  # for aurac
        self.all_e2e_latencies = [] # for end-to-end latency
        self.all_init_infer_latencies = [] # for init inference latency
        self.all_total_output_tokens = [] # for total output tokens

    def _eval_ans_correct(self, idx, log_dict):
        # Evaluate LVLM's answer
        flag_ans_correct = True
        label = log_dict[idx]['benchmark']['label']
        ans = log_dict[idx]['acc']['ans']
        if BENCHMARK_TYPE[self.args.benchmark] == 'MULTI_CHOICE':
            flag_ans_correct = str(label) in ans
        log_dict[idx]['acc']['flag_ans_correct'] = flag_ans_correct

    def _compute_aurac(self, uncertainties, correctness):
        """
        Compute Area Under the Rejection-Accuracy Curve (AURAC).
        
        Args:
            uncertainties (list or np.ndarray): Uncertainty scores (higher = more uncertain).
            correctness (list or np.ndarray): Binary correctness labels (1 = correct, 0 = wrong).
            num_points (int): Number of rejection thresholds to evaluate (default: 100).
        
        Returns:
            float: AURAC value in [0,1].
        """
        uncertainties = np.array(uncertainties)
        correctness = np.array(correctness)

        # Sort by uncertainty descending (reject highest first)
        sorted_idx = np.argsort(-uncertainties)
        correctness_sorted = correctness[sorted_idx]

        n = len(correctness)
        rejection_rates = np.linspace(0, 1, self.num_points)
        accuracies = []

        for r in rejection_rates:
            k = int(r * n)  # number of rejected samples
            kept = correctness_sorted[k:]
            if len(kept) > 0:
                acc = np.mean(kept)
            else:
                acc = 0.0
            accuracies.append(acc)

        # Trapezoidal integration
        aurac = np.trapz(accuracies, rejection_rates)
        return aurac

    def calculate_single(self, args, idx, log_dict):
        # Accuracy
        self._eval_ans_correct(idx, log_dict)
        self.total_sample += 1
        if log_dict[idx]['acc']['flag_ans_correct']:
            self.cnt_correct_ans += 1

        # AUROC
        self.uncertainty_scores.append(log_dict[idx]['uq']['normalized_uncertainty'])
        label = 0 if log_dict[idx]['acc']['flag_ans_correct'] else 1 # label=0: lvlm ans correct
        self.auroc_labels.append(label)

        # AURAC
        correctness = 1 if log_dict[idx]['acc']['flag_ans_correct'] else 0
        self.correctness_labels.append(correctness)
        
        # Latency
        end_to_end_latency = 0.0
        end_to_end_latency = log_dict[idx]['latency']['init_infer_latency']
        self.all_init_infer_latencies.append([log_dict[idx]['latency']['init_infer_latency']])
        for latency in log_dict[idx]['latency']['sampling_latency_list']:
            end_to_end_latency += latency
        end_to_end_latency += log_dict[idx]['latency']['uq_latency']
        log_dict[idx]['latency']['end_to_end_latency'] = end_to_end_latency
        self.all_e2e_latencies.append(end_to_end_latency)
        
        # Output tokens
        total_output_tokens = 0
        total_output_tokens = log_dict[idx]['output_tokens']['init_output_tokens']
        for tokens in log_dict[idx]['output_tokens']['output_tokens_sampling_list']:
            total_output_tokens += tokens
        log_dict[idx]['output_tokens']['total_output_tokens'] = total_output_tokens
        self.all_total_output_tokens.append(total_output_tokens)

    def calculate_all(self):
        # Calculate benchmark accuracy
        accuracy = (self.cnt_correct_ans / self.total_sample) * 100 if self.total_sample > 0 else 0.0

        # Calculate auroc
        auroc = roc_auc_score(self.auroc_labels, self.uncertainty_scores)

        # Calculate aurac
        aurac = self._compute_aurac(
            self.uncertainty_scores,
            self.correctness_labels 
        )
        
        # Calculate stats of initial inference latency
        init_infer_latency_second_stats = {
            'mean': float(np.mean(self.all_init_infer_latencies)),
            'median': float(np.median(self.all_init_infer_latencies)),
            'min': float(np.min(self.all_init_infer_latencies)),
            'max': float(np.max(self.all_init_infer_latencies))
        }

        # Calculate stats of e2e latency
        e2e_latency_second_stats = {
            'mean': float(np.mean(self.all_e2e_latencies)),
            'median': float(np.median(self.all_e2e_latencies)),
            'p50': float(np.percentile(self.all_e2e_latencies, 50)),
            'p95': float(np.percentile(self.all_e2e_latencies, 95)),
            'p99': float(np.percentile(self.all_e2e_latencies, 99)),
            'min': float(np.min(self.all_e2e_latencies)),
            'max': float(np.max(self.all_e2e_latencies))
        }

        # Calculate stats of total output tokens
        total_output_tokens_stats = {
            'mean': int(np.mean(self.all_total_output_tokens)),
            'median': int(np.median(self.all_total_output_tokens)),
            'min': int(np.min(self.all_total_output_tokens)),
            'max': int(np.max(self.all_total_output_tokens))
        }

        result = {
            'total_sample': self.total_sample,
            'accuracy': accuracy,
            'auroc': auroc,
            'aurac': aurac,
            'init_infer_latency_second_stats': init_infer_latency_second_stats,
            'e2e_latency_second_stats': e2e_latency_second_stats,
            'total_output_tokens_stats': total_output_tokens_stats
        }

        return result
