
import re
import collections
import math
import time
# --- Import local modules ---
from uq.semantic_entropy import *
# ----------------------------


BENCHMARK_TYPE = {
    'MMMU': 'MULTI_CHOICE',
    'ScienceQA': 'MULTI_CHOICE',
    'MMBench': 'MULTI_CHOICE',
}

EPSILON = 1e-6  # avoid zero-division and log(0)

def sample_lvlm_ans(args, lvlm, sample, log_dict):
    # Infer LVLM answer
    ans, ans_logprob, infer_latency, output_tokens = lvlm.generate(
        sample['img'],
        sample['question'],
        args.sampling_temp
    )

    # Log
    log_dict[sample['idx']]['uq']['sampled_ans_list'].append(ans)
    log_dict[sample['idx']]['latency']['sampling_latency_list'].append(infer_latency)
    log_dict[sample['idx']]['output_tokens']['output_tokens_sampling_list'].append(output_tokens)

def estimate_uncertainty(args, lvlm, sample, log_dict):
    start_time = time.time()
    ans_cluster_idx_list = []
    sampled_ans_list = log_dict[sample['idx']]['uq']['sampled_ans_list']

    # Cluster answers
    if BENCHMARK_TYPE[args.benchmark] == 'MULTI_CHOICE':
        for ans in sampled_ans_list:
            if re.search(r'\d+', ans) is None or int(re.search(r'\d+', ans).group()) >= sample['num_c']:
                ans_cluster_idx_list.append(-1)
            else:
                ans_cluster_idx_list.append(int(re.search(r'\d+', ans).group()))
    
    log_dict[sample['idx']]['uq']['ans_cluster_idx_list'] = ans_cluster_idx_list
    
    # Cluster dictionary
    cluster_dict = collections.Counter(ans_cluster_idx_list)
    log_dict[sample['idx']]['uq']['cluster_dict'] = cluster_dict

    # Raw entropy
    raw_entropy = -sum((cnt / args.sampling_time) * math.log2(cnt / args.sampling_time) for cnt in cluster_dict.values())
    log_dict[sample['idx']]['uq']['uncertainty'] = raw_entropy

    # Normalized entropy
    cluster_dict_size = len(cluster_dict)
    normalized_entropy = raw_entropy / (math.log2(cluster_dict_size) + EPSILON)
    log_dict[sample['idx']]['uq']['normalized_uncertainty'] = normalized_entropy

    # Calculate estimation latency
    uq_latency = time.time() - start_time
    log_dict[sample['idx']]['latency']['uq_latency'] = uq_latency

def route_uq_method(args, lvlm, sample, log_dict):
    log_dict[sample['idx']]['uq']['sampled_ans_list'] = []

    # Different UQ methods, different sampling ans strategies
    if args.uq_method == 'semantic_entropy':
        semantic_entropy(args, lvlm, sample, log_dict)
        return estimate_uncertainty(args, lvlm, sample, log_dict)
    else:
        raise ValueError(f"Unknown UQ method: {args.uq_method}")
    
