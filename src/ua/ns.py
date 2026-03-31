import math
import time
from typing import Any, Dict, Tuple, List


class NaiveSelection:
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def _convert_to_probability_vector(self, cluster_dict: Dict[Any, int]) -> Dict[Any, float]:
        """Normalize counts to probabilities (no alignment)."""
        total = sum(cluster_dict.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in cluster_dict.items()}

    def _calculate_shannon_entropy(self, prob_vector: Dict[Any, float]) -> float:
        """Base-2 Shannon entropy (raw, unaligned)."""
        h = 0.0
        for p in prob_vector.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def fuse_per_sample(self, sample_result: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Choose the model with the lowest entropy and adopt its answer and entropy."""
        start = time.time()

        try:
            # --- Dynamically determine number of logs ---
            log_keys = [k for k in sample_result if k.startswith('log') and k.endswith('_data')]
            num_logs = len(log_keys)
            
            local_probs = []
            local_entropies = []

            # --- Loop to get all local vectors and entropies ---
            for i in range(num_logs):
                log_name = f'log{i+1}_data'
                raw_dict = sample_result[log_name]["raw_cluster_dict"]

                # Convert to prob vectors
                prob_vector = self._convert_to_probability_vector(raw_dict)
                local_probs.append(prob_vector)

                # Compute raw entropies
                entropy = self._calculate_shannon_entropy(prob_vector)
                local_entropies.append(entropy)
            # ---------------------------------------------------
            
            if not local_entropies:
                raise ValueError("No valid log data found to compute entropies.")

            # --- Pick model with lowest entropy (more confident) ---
            chosen_index = min(range(num_logs), key=local_entropies.__getitem__)
            
            chosen_prob_vector = local_probs[chosen_index]
            chosen_src = f"log{chosen_index + 1}"
            chosen_entropy = local_entropies[chosen_index]
            # -------------------------------------------------------

            # Predicted answer: argmax probability
            # Handle empty vector case
            if not chosen_prob_vector:
                fused_ans = None
                fused_conf = 0.0
            else:
                fused_ans = max(chosen_prob_vector, key=chosen_prob_vector.get)
                fused_conf = chosen_prob_vector[fused_ans]

            # --- Store results (dynamic) ---
            sample_result["fusion_results"] = {
                "chosen_model": chosen_src,
                "fused_probability_vector": chosen_prob_vector, # for align eval_elop.py
                "fused_ans": fused_ans,
                "fused_confidence": fused_conf,
                "fused_entropy": chosen_entropy,
            }
            
            # Add all individual log entropies
            for i in range(num_logs):
                sample_result["fusion_results"][f'log{i+1}_entropy'] = local_entropies[i]
            # --------------------------------

        except Exception as e:
            sample_result["error"] = str(e)

        latency = time.time() - start
        return sample_result, latency