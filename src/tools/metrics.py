import torch
import numpy as np


def eval_explain_ratio_given_indices(CI, masks, selected_indices):
    if len(selected_indices) == 0: return 0.0

    selected = torch.zeros(CI.shape[0]).bool()
    selected[selected_indices] = True

    if isinstance(CI, np.ndarray): CI = torch.FloatTensor(CI.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())

    not_empty = torch.any(masks, dim=1)
    unselected = torch.logical_not(selected)

    numerator = torch.abs(CI[selected][not_empty[selected]]).sum()
    denominator = torch.abs(CI[selected][not_empty[selected]]).sum() + \
                  torch.abs(CI[unselected][not_empty[unselected]].sum()) + 1e-7
    ratio = numerator / denominator

    return ratio.item()