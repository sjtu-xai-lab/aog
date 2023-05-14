import torch
import numpy as np
from .and_harsanyi_utils import get_Iand2reward_mat


def eval_explain_ratio(interactions, masks, selected):
    """
    Evaluate the explanation ratio when using selected interactions to explain a deep model
    :param interactions: [2^n,] float tensor, the interaction of each pattern
    :param masks: [2^n, n] bool tensor, interaction patterns
    :param selected: [2^n] bool tensor, indicating whether each interaction pattern is selected
    :return: the explanation ratio
    """
    if torch.sum(selected) == 0: return 0.0

    if isinstance(interactions, np.ndarray): interactions = torch.FloatTensor(interactions.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())
    if isinstance(selected, np.ndarray): selected = torch.BoolTensor(selected.copy())

    not_empty = torch.any(masks, dim=1)
    unselected = torch.logical_not(selected)

    numerator = torch.abs(interactions[selected][not_empty[selected]]).sum()
    denominator = torch.abs(interactions[selected][not_empty[selected]]).sum() + \
                  torch.abs(interactions[unselected][not_empty[unselected]].sum()) + 1e-7
    ratio = numerator / denominator

    return ratio.item()


def remove_noisy_greedy(
        rewards: torch.Tensor,
        interactions: torch.Tensor,
        masks: torch.BoolTensor,
        min_patterns: int = 15,
        max_patterns: int = 80,
        n_greedy: int = 20,
        thres_square_error: float = 0.1,
        thres_explain_ratio: float = 0.95
):

    device = rewards.device
    n_dim = masks.shape[1]
    harsanyi2reward = get_Iand2reward_mat(n_dim).float().to(device)

    original_interactions = interactions.clone()
    interactions = interactions.clone()
    v_N = rewards[torch.all(masks, dim=1)].item()
    v_empty = rewards[torch.logical_not(torch.any(masks, dim=1))].item()
    order = torch.argsort(torch.abs(interactions)).tolist()  # from low-strength to high-strength

    n_all_patterns = interactions.shape[0]
    unremoved = torch.ones(n_all_patterns, dtype=torch.bool, device=device)

    removed_coalition_ids = []
    square_errors = []
    explain_ratios = []

    for _ in range(n_all_patterns - min_patterns):
        candidates = order[:n_greedy]
        to_remove = candidates[0]
        interactions_ = interactions.clone()
        interactions_[to_remove] = 0.
        square_error = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, interactions_)))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            interactions_ = interactions.clone()
            interactions_[candidate] = 0.
            square_error_ = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, interactions_)))
            if square_error_ < square_error:
                to_remove = candidate
                square_error = square_error_

        interactions[to_remove] = 0.
        order.remove(to_remove)
        unremoved[to_remove] = False
        removed_coalition_ids.append(to_remove)
        square_errors.append(square_error.item())
        explain_ratio = eval_explain_ratio(original_interactions, masks, unremoved)
        explain_ratios.append(explain_ratio)

    assert len(square_errors) == len(removed_coalition_ids)
    assert len(square_errors) == len(explain_ratios)
    square_errors = np.array(square_errors)
    explain_ratios = np.array(explain_ratios)

    normalized_errors = square_errors / (torch.sum(rewards * rewards).item() + 1e-7)

    satisfy = np.logical_and(explain_ratios > thres_explain_ratio, normalized_errors < thres_square_error)
    last_satisfy = 0
    for i in range(len(satisfy) - 1, -1, -1):
        if satisfy[i]:
            last_satisfy = i
            break

    if n_all_patterns - last_satisfy - 1 > max_patterns:
        last_satisfy = n_all_patterns - max_patterns - 1

    final_removed_ids = removed_coalition_ids[:last_satisfy+1]
    final_retained_ids = [idx for idx in range(interactions.shape[0]) if idx not in final_removed_ids]
    final_normalized_error = normalized_errors[last_satisfy]
    final_ratio = explain_ratios[last_satisfy]
    print(f"Removing noisy patterns "
          f"-- # coalitions: {len(final_retained_ids)} "
          f"| normalized error: {final_normalized_error:.4f} "
          f"| explain ratio: {final_ratio:.4f}\n")

    result_dict = {
        "unremoved": unremoved,
        "errors": square_errors,
        "explain_ratios": explain_ratios,
        "final_retained": final_retained_ids,
        "final_error": final_normalized_error,
        "final_ratio": final_ratio
    }
    return result_dict
