import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.append("..")
from harsanyi.interaction_utils import generate_all_masks, is_A_subset_B, select_subset, \
    set_to_index, get_subset, is_A_subset_Bs
from pprint import pprint


def calculate_total_code_length(patterns, eval_val):
    eps = 1e-8
    frequency = np.matmul(eval_val, patterns)
    frequency /= frequency.sum()
    codeword_length = np.log(1 / (frequency + eps))
    total_code_length = np.dot(np.matmul(eval_val, patterns), codeword_length)
    return total_code_length


def calculate_average_code_length(patterns, eval_val):
    eps = 1e-8
    frequency = np.matmul(eval_val, patterns)
    frequency /= frequency.sum()
    codeword_length = np.log(1 / (frequency + eps))
    average_code_length = np.dot(frequency, codeword_length)
    return average_code_length


def calculate_codebook_length(
        original_patterns: np.ndarray,  # [n_patterns, n_attributes]
        merged_patterns: np.ndarray,  # [n_patterns, n_merged]
        codebook: np.ndarray,  # [n_merged, n_attributes]
        eval_val: np.ndarray  # [n_patterns,]
):
    eps = 1e-8
    is_codeword_used = np.any(merged_patterns, axis=0)  # whether the merged patterns are used [n_merged, n_attributes]
    # 1. the coding length of the itemset
    item_frequency = np.matmul(eval_val, original_patterns)
    item_frequency /= item_frequency.sum()
    item_code_length = np.log(1 / (item_frequency + eps))  # [n_attributes,]
    itemset_code_length = np.matmul(codebook, item_code_length)
    itemset_code_length = itemset_code_length[is_codeword_used].sum()
    # 2. the coding length of codewords
    codeword_frequency = np.matmul(eval_val, merged_patterns)
    codeword_frequency /= codeword_frequency.sum()
    codewords_code_length = np.log(1 / (codeword_frequency + eps))[is_codeword_used].sum()
    # 3. the coding length of the codebook is the sum of (1) and (2)
    codebook_complexity = itemset_code_length + codewords_code_length
    return codebook_complexity


def aggregate_pattern_iterative(
        patterns: np.ndarray,
        interactions: np.ndarray,
        max_iter: int = 5,
        entropy_lamb: float = 10.0,
        early_stop: bool = True
):
    '''
    This function will aggregate feature dimensions to form new 'code words', to shorten the code length.
    :param patterns: <numpy.ndarray> (n_patterns, n_features) -- an array of patterns (masks)
    :param interactions: <numpy.ndarray> (n_patterns, ) -- the interaction of these patterns
    :param max_iter: the maximum number of merges
    :param objective: str -- "total_length" or "avg_length"  "10.0_entropy+total_length-eff-early"
    :return: the merged codewords, the new concept mask, code length during optimization
    '''
    if isinstance(patterns, torch.Tensor):
        patterns = patterns.cpu().numpy()
    if isinstance(interactions, torch.Tensor):
        interactions = interactions.cpu().numpy()

    strengths = np.abs(interactions.copy())
    original_patterns = patterns.copy()
    n_feature_dim = patterns.shape[1]
    codebook = np.eye(n_feature_dim).astype(bool)

    calculate_code_length = lambda ori_patterns, mer_patterns, cdbk, val: \
        calculate_total_code_length(mer_patterns, val) \
        + entropy_lamb * calculate_average_code_length(mer_patterns, val)

    code_lengths = {}

    code_length = calculate_code_length(original_patterns, patterns, codebook, strengths)

    code_lengths["sum"] = [code_length]
    code_lengths["words"] = [calculate_total_code_length(patterns, strengths)]
    code_lengths["entropy"] = [calculate_average_code_length(patterns, strengths)]
    code_lengths["decrease-per-dim"] = []
    code_lengths["merge-size"] = []

    for _ in range(max_iter):
        length_after_merge = {}

        for concept in patterns:
            sub_patterns = get_subset(concept)
            for sub_concept in sub_patterns:
                if sub_concept.sum() < 2: continue  # if aiming to merge 0 or 1 code words ...
                sub_concept_idx = set_to_index(sub_concept)
                if sub_concept_idx in length_after_merge.keys(): continue
                # judge if each concept in 'patterns' has such combination
                flag = is_A_subset_Bs(sub_concept, patterns)

                patterns_after_merge = patterns.copy()
                indice = np.outer(flag, sub_concept).astype(bool)
                patterns_after_merge[indice] = False
                patterns_after_merge = np.hstack([patterns_after_merge, flag.reshape(-1, 1)])
                codebook_after_merge = np.vstack([codebook, np.any(codebook[sub_concept], axis=0)])

                code_length = calculate_code_length(original_patterns, patterns_after_merge, codebook_after_merge, strengths)
                length_after_merge[sub_concept_idx] = [code_length, sub_concept, patterns_after_merge, codebook_after_merge]

        if len(length_after_merge.keys()) == 0: break

        # select the merge
        prev_code_length = code_lengths["sum"][-1]
        code_length, _, patterns_new, _ = sorted(
            list(length_after_merge.values()), key=lambda item: (item[0] - prev_code_length) / np.sum(item[3][-1]),
        )[0]
        if early_stop and code_length > code_lengths["sum"][-1]: break
        if np.sum(patterns_new[:, -1]) == 1: break  # the merged pattern is actually not common
        code_length, _, patterns, codebook = sorted(
            list(length_after_merge.values()), key=lambda item: (item[0] - prev_code_length) / np.sum(item[3][-1]),
        )[0]
        code_lengths["sum"].append(code_length)

        # update the saved data
        code_lengths["words"].append(calculate_total_code_length(patterns, strengths))
        code_lengths["entropy"].append(calculate_average_code_length(patterns, strengths))
        decrease_per_dim = (code_lengths["sum"][-2] - code_lengths["sum"][-1]) / np.sum(codebook[-1])
        code_lengths["decrease-per-dim"].append(decrease_per_dim)
        code_lengths["merge-size"].append(np.sum(codebook[-1]))

    return codebook[original_patterns.shape[1]:], patterns, code_lengths


def calculate_edge_num(merged_patterns, aggregated_patterns):
    if aggregated_patterns.shape[0] == 0:
        return 0
    n_feature_dim = merged_patterns.shape[1]
    codebook = np.vstack([np.eye(n_feature_dim).astype(bool), merged_patterns])
    edge_num = np.any(aggregated_patterns, axis=1).sum()
    edge_num += aggregated_patterns.sum()
    is_code_used = np.any(aggregated_patterns, axis=0)
    is_code_used[:n_feature_dim] = np.logical_or(
        is_code_used[:n_feature_dim], np.any(merged_patterns[is_code_used[n_feature_dim:]], axis=0)
    )
    # edge_num += codebook[n_feature_dim:][is_code_used[n_feature_dim:]].sum()
    edge_num += merged_patterns[is_code_used[n_feature_dim:]].sum()
    return edge_num


def calculate_node_num(merged_patterns, aggregated_patterns):
    if aggregated_patterns.shape[0] == 0:
        return 0
    n_feature_dim = merged_patterns.shape[1]
    codebook = np.vstack([np.eye(n_feature_dim).astype(bool), merged_patterns])
    node_num = 1 + np.any(aggregated_patterns, axis=1).sum()
    is_code_used = np.any(aggregated_patterns, axis=0)
    is_code_used[:n_feature_dim] = np.logical_or(
        is_code_used[:n_feature_dim], np.any(merged_patterns[is_code_used[n_feature_dim:]], axis=0)
    )
    node_num += is_code_used.sum()
    return node_num
