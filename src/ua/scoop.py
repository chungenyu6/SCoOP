import math
import time
from typing import Any, Dict, List, Tuple


class SCoOP:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def _convert_to_probability_vector(self, cluster_dict: Dict[Any, int]) -> Dict[Any, float]:
        """Converts a cluster count dictionary to a probability vector."""
        total_count = sum(cluster_dict.values())
        if total_count == 0:
            return {}

        prob_vector = {}
        for cluster, count in cluster_dict.items():
            prob_vector[cluster] = count / total_count 

        return prob_vector
    
    def _standardize_to_full_vector(
        self, prob_vectors: List[Dict[Any, float]], unified_choices: List[Any] = None
    ) -> Tuple[List[Dict[Any, float]], List[Any]]:
        """Align partial probability vectors to full standardized vectors."""
        if unified_choices is None:
            unified_choices = set()
            for pv in prob_vectors:
                unified_choices.update(pv.keys())
            unified_choices = sorted(list(unified_choices))

        standardized_vectors = []
        for prob_vector in prob_vectors:
            full_vector = {}
            for choice in unified_choices:
                full_vector[choice] = prob_vector.get(choice, 0.0)
            standardized_vectors.append(full_vector)

        return standardized_vectors, unified_choices

    def _calculate_shannon_entropy(self, prob_vector: Dict[Any, float]) -> float:
        """Calculates the Shannon entropy for a probability vector."""
        entropy = 0.0
        for prob in prob_vector.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def _calculate_entropy_weights(self, entropies: List[float]) -> List[float]:
        """Calculate weights based on inverse entropy (with epsilon regularization)."""
        # Compute raw confidences (inverse entropy)
        raw_confidences = [1.0 / (entropy + self.epsilon) for entropy in entropies]

        # Normalize to sum to 1
        total_confidence = sum(raw_confidences)
        if total_confidence == 0: # Avoid division by zero if all entropies are huge
             return [1.0 / len(entropies)] * len(entropies)
        weights = [conf / total_confidence for conf in raw_confidences]

        return weights

    def _linear_opinion_pooling(
        self,
        standardized_vectors: List[Dict[Any, float]],
        weights: List[float],
        unified_choices: List[Any],
    ) -> Dict[Any, float]:
        """Perform weighted arithmetic mean fusion (Linear Opinion Pooling)."""
        fused_vector = {}
        for choice in unified_choices:
            fused_prob = 0.0
            for i, vector in enumerate(standardized_vectors):
                fused_prob += weights[i] * vector[choice]
            fused_vector[choice] = fused_prob
        return fused_vector
    
    def fuse_per_sample(self, sample_result: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Process a specific sample from N log files.
        (Logic updated for N logs)
        """
        # --- Dynamically determine number of logs ---
        log_keys = [k for k in sample_result if k.startswith('log') and k.endswith('_data')]
        num_logs = len(log_keys)
        
        try:
            start_time = time.time()  # Fusion time start after extraction

            # --- Dynamic Extraction ---
            raw_clusters = []
            for i in range(num_logs):
                raw_clusters.append(sample_result[f'log{i+1}_data']['raw_cluster_dict'])
            
            # Convert to probability vectors
            prob_vectors = [self._convert_to_probability_vector(clus) for clus in raw_clusters]
            
            # Store back in sample_result
            for i in range(num_logs):
                sample_result[f'log{i+1}_data']['probability_vector'] = prob_vectors[i]
            # --------------------------

            # Standardize to full vectors
            standardized_vectors, unified_choices = self._standardize_to_full_vector(prob_vectors)

            # --- Dynamic Entropy Calculation ---
            if not standardized_vectors or not standardized_vectors[0]:
                # Handle case with no choices (e.g., empty cluster dicts)
                n = 1 
            else:
                n = len(standardized_vectors[0])
                
            entropies = [self._calculate_shannon_entropy(vec) for vec in standardized_vectors]
            log_n_eps = math.log2(n) + self.epsilon
            if log_n_eps == 0: log_n_eps = self.epsilon # Safety for n=1
            
            normalized_entropies = [e / log_n_eps for e in entropies]
            # -----------------------------------

            # Calculate weights
            weights = self._calculate_entropy_weights(normalized_entropies)

            # Perform fusion
            fused_vector = self._linear_opinion_pooling(standardized_vectors, weights, unified_choices)
            fused_entropy_raw = self._calculate_shannon_entropy(fused_vector)
            m = len(fused_vector) if fused_vector else 1
            log_m_eps = math.log2(m) + self.epsilon
            if log_m_eps == 0: log_m_eps = self.epsilon # Safety for m=1
            fused_entropy = fused_entropy_raw / log_m_eps # normalize

            # --- Dynamic tie-break ---
            max_prob = max(fused_vector.values()) if fused_vector else 0
            candidates = [c for c, v in fused_vector.items() if v == max_prob]

            if len(candidates) == 1:
                fused_ans = candidates[0]
            elif not candidates:
                 fused_ans = None # Handle empty fusion
            else:
                # prefer model with lower entropy
                min_entropy = min(normalized_entropies)
                # Find all indices with the minimum entropy
                min_entropy_indices = [i for i, e in enumerate(normalized_entropies) if e == min_entropy]
                
                # Tie-break 1: Pick the one with the lowest index (e.g., log1 over log2)
                chosen_index = min_entropy_indices[0]
                chosen_std = standardized_vectors[chosen_index]

                # among tied candidates, pick the one with higher prob in chosen_std
                best_val = max(chosen_std.get(c, 0.0) for c in candidates)
                best_candidates = [c for c in candidates if chosen_std.get(c, 0.0) == best_val]

                # deterministic fallback: lexicographic order
                fused_ans = sorted(best_candidates)[0]
            # -----------------------------

            # Store fusion results
            sample_result['fusion_results'] = {
                'unified_choices': unified_choices,
                'fused_probability_vector': fused_vector,
                'fused_entropy': fused_entropy,
                'fused_ans': fused_ans
            }
            
            # --- Dynamically add log-specific results ---
            for i in range(num_logs):
                log_num = i + 1
                sample_result['fusion_results'][f'log{log_num}_standardized'] = standardized_vectors[i]
                sample_result['fusion_results'][f'log{log_num}_entropy'] = entropies[i]
                sample_result['fusion_results'][f'normalized_log{log_num}_entropy'] = normalized_entropies[i]
                sample_result['fusion_results'][f'log{log_num}_weight'] = weights[i]
            # ---------------------------------------------

            # Calculate fusion latency
            end_time = time.time()
            fusion_latency = end_time - start_time

        except Exception as e:
            sample_result['error'] = str(e)
            fusion_latency = 0.0

        return sample_result, fusion_latency