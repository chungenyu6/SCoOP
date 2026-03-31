import math
import time
from typing import Any, Dict, Tuple, List, Optional


class MajorityVoting:
    """
    Implements discrete majority voting (p_MV) based on local probabilities.

    This method computes the local probability vector (p_k) for each model
    from its raw counts. It finds the top-1 vote (theta_k*) from this
    local vector. It then creates a histogram of these discrete votes,
    normalizes by the number of models (K) to get the p_MV distribution,
    and finds the final prediction.

    Tie-breaking is handled by selecting the class with the highest
    local probability (p_k(theta_k*)) among the tied candidates.
    """
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def _convert_to_probability_vector(self, cluster_dict: Dict[Any, int]) -> Dict[Any, float]:
        """
        Normalize counts to probabilities.
        Computes p(theta) = N(theta) / N_tot.
        """
        total = sum(cluster_dict.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in cluster_dict.items()}

    def _get_top_1(self, prob_vector: Dict[Any, float]) -> Tuple[Optional[Any], float]:
        """
        Finds the top-1 class and its confidence (probability).
        Returns (theta_k*, s_k)
        """
        if not prob_vector:
            return None, 0.0

        # Find the class with the highest probability
        top_class = max(prob_vector, key=prob_vector.get)
        # Get the confidence for that class
        confidence = prob_vector[top_class]

        return top_class, confidence

    def _calculate_shannon_entropy(self, prob_vector: Dict[Any, float]) -> float:
        """Base-2 Shannon entropy."""
        h = 0.0
        for p in prob_vector.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def fuse_per_sample(self, sample_result: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Perform discrete majority voting (p_MV) based on local p_k."""
        start = time.time()

        try:
            # --- Dynamically determine number of logs ---
            log_keys = [k for k in sample_result if k.startswith('log') and k.endswith('_data')]
            num_logs = len(log_keys) # This is K

            local_probs = []
            local_entropies = []
            top_votes_list = [] # For storing results
            
            # --- NEW: Stores (theta_k*, p_k(theta_k*)) for tie-breaking ---
            votes_for_tiebreaking: List[Tuple[Any, float]] = []
            
            # --- NEW: Initialize dictionary for discrete vote counts ---
            mv_counts: Dict[Any, int] = {}
            
            # --- Step 1: Loop to get top-1 vote from each model's p_k ---
            for i in range(num_logs):
                log_name = f'log{i+1}_data'
                
                # 1. Get raw cluster data
                raw_dict = sample_result[log_name]["raw_cluster_dict"]
                
                # 2. Convert to local probability vector (p_k)
                prob_vector = self._convert_to_probability_vector(raw_dict)
                local_probs.append(prob_vector)
                
                # (Log local entropy)
                entropy = self._calculate_shannon_entropy(prob_vector)
                local_entropies.append(entropy)

                # 3. Find top-1 class and score (theta_k*, p_k(theta_k*))
                top_class, top_score = self._get_top_1(prob_vector)
                
                # (Store for logging)
                top_votes_list.append({"class": top_class, "score": top_score})
                
                if top_class is not None:
                    # 4. Aggregate discrete vote (for p_MV)
                    mv_counts[top_class] = mv_counts.get(top_class, 0) + 1
                    
                    # 5. Store score (for tie-breaking)
                    votes_for_tiebreaking.append((top_class, top_score))
            # ------------------------------------------------------------------

            # 6. Normalize vote counts to get p_MV
            # p_MV(theta) = (1/K) * sum(1[theta_k* == theta])
            fused_prob_vector = self._convert_to_probability_vector(mv_counts)
            
            # 7. Final Prediction (theta*) with tie-breaking
            fused_ans = None
            if fused_prob_vector:
                # Find max probability (i.e., max vote count)
                max_prob = max(fused_prob_vector.values())
                # Get all candidates with that probability
                candidates = [
                    c for c, p in fused_prob_vector.items() if p == max_prob
                ]
                
                if len(candidates) == 1:
                    fused_ans = candidates[0]
                else:
                    # --- TIE-BREAKING LOGIC ---
                    # Find the highest local prob (p_k) among tied classes
                    best_score = -1.0
                    fused_ans = None # Will be set to the winner
                    
                    # Sort candidates for deterministic result if scores *also* tie
                    for cls in sorted(candidates):
                        # Find the max score *any* model gave to this class
                        # (from models that voted for it)
                        max_score_for_cls = max(
                            score for v_cls, score in votes_for_tiebreaking if v_cls == cls
                        )
                        
                        if max_score_for_cls > best_score:
                            best_score = max_score_for_cls
                            fused_ans = cls
                    # ------------------------------

            # 8. Uncertainty (H_MV_norm)
            h_raw = self._calculate_shannon_entropy(fused_prob_vector) # H_MV

            # Union of all classes seen by any model
            union_classes = set()
            for probs in local_probs:
                union_classes.update(probs.keys())
            J_union = len(union_classes)

            # Normalize entropy by log2(J_union)
            h_norm = h_raw / (math.log2(J_union) + self.epsilon)

            # --- Store results ---
            sample_result["fusion_results"] = {
                "majority_vote_counts": mv_counts,
                "J_union": J_union,
                # This is now p_MV
                "fused_probability_vector": fused_prob_vector,
                # This is now H_MV_norm
                "fused_entropy": h_norm,
                # This is now theta* (with new tie-break)
                "fused_ans": fused_ans,
            }

            # Add log-specific info
            for i in range(num_logs):
                log_num = i + 1
                sample_result["fusion_results"][f'log{log_num}_local_prob_vector'] = local_probs[i]
                sample_result["fusion_results"][f'log{log_num}_entropy'] = local_entropies[i]
                sample_result["fusion_results"][f'log{log_num}_vote'] = top_votes_list[i]
            # --------------------------------

        except Exception as e:
            sample_result["error"] = str(e)

        latency = time.time() - start
        return sample_result, latency