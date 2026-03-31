# --- Import local modules ---
from uq import estimate # to avoid circular import
# ----------------------------


def semantic_entropy(args, lvlm, sample, log_dict):
    # Sample answers from LVLM
    log_dict[sample['idx']]['uq']['sampled_ans_list'] = []
    log_dict[sample['idx']]['latency']['sampling_latency_list'] = []
    log_dict[sample['idx']]['output_tokens']['output_tokens_sampling_list'] = []    
    for _ in range(args.sampling_time):
        estimate.sample_lvlm_ans(args, lvlm, sample, log_dict)
