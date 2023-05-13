import torch
from tqdm import tqdm
import numpy as np

from typing import Callable, List, Tuple, Union, Dict
import sys

from .interaction_utils import generate_all_masks, generate_subset_masks


def get_reward2Iand_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    # for i in tqdm(range(n_masks), ncols=100, desc="Generating mask"):
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Iand2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def _train_baseline(
        f_baseline2rewards: Callable,
        f_masksbaseline2rewards: Callable,
        baseline_init: torch.Tensor,
        loss_type: str,
        baseline_lr: float,
        niter: int,
        baseline_min: Union[float, torch.Tensor],
        baseline_max: Union[float, torch.Tensor],
        reward2Iand: torch.Tensor,
        calc_bs: int = None,
        verbose: int = 1
):
    device = baseline_init.device
    if isinstance(baseline_min, float):
        baseline_min = torch.ones_like(baseline_init) * baseline_min
    if isinstance(baseline_max, float):
        baseline_max = torch.ones_like(baseline_init) * baseline_max

    assert torch.all(baseline_max >= baseline_min), f"[baseline setting error] max < min, please check," \
                                                    f" max: {baseline_max}," \
                                                    f" min: {baseline_min}"

    with torch.no_grad():
        masks, rewards = f_baseline2rewards(baseline_init)
        Iand = reward2Iand @ rewards

    if calc_bs is None:
        calc_bs = rewards.shape[0]

    # set up learning rate
    log_baseline_lr = np.log10(baseline_lr)
    eta_baseline_list = np.logspace(log_baseline_lr, log_baseline_lr - 1, niter)

    # initialize baseline value
    baseline = baseline_init.clone().requires_grad_(True)

    if loss_type == "l1":
        losses = {"loss": []}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")
    progresses = {
        "I_and": [Iand.data.clone().cpu().numpy()],
        "baseline": [baseline.data.clone().cpu().numpy()]
    }

    pbar_iter = tqdm(range(niter), ncols=100, desc="Optimizing b")

    for it in pbar_iter:
        # loss = rewards * reward_coefs
        if loss_type == "l1":
            with torch.no_grad():
                reward_coefs = torch.matmul(torch.sign(Iand), reward2Iand)
        else:
            raise NotImplementedError(f"Loss type {loss_type} unrecognized."
                                      f"You can manually define this loss function.")

        indices = list(range(masks.shape[0]))

        baseline_grad = 0.
        total_loss_baseline = 0.

        if verbose == 1:
            pbar_batch = tqdm(range(int(np.ceil(len(indices) / calc_bs))), desc="Optimizing b", ncols=100)
        else:
            pbar_batch = range(int(np.ceil(len(indices) / calc_bs)))

        for batch_id in pbar_batch:
            batch_indices = indices[batch_id*calc_bs : batch_id*calc_bs+calc_bs]
            _, rewards = f_masksbaseline2rewards(masks[batch_indices], baseline)

            loss_baseline = torch.matmul(reward_coefs[batch_indices], rewards)

            grad = torch.autograd.grad(loss_baseline, baseline, only_inputs=True)[0]
            baseline_grad += grad.data.clone()
            total_loss_baseline += loss_baseline.item()

        baseline.data = baseline.data - eta_baseline_list[it] * baseline_grad
        # baseline.data = torch.clamp(baseline.data, baseline_min, baseline_max)
        baseline = torch.max(torch.min(baseline.data, baseline_max), baseline_min)\
            .clone().detach().requires_grad_(True).float()

        if verbose == 1 and loss_type == "l1":
            print("loss:", total_loss_baseline, "| L1:", torch.sum(torch.abs(Iand)))

        # update the rewards, note that rewards change after the optimization of baseline values
        with torch.no_grad():
            masks, rewards = f_baseline2rewards(baseline)
            Iand = reward2Iand @ rewards

        # update loss
        losses["loss"].append(total_loss_baseline)

        # update intermediate results
        baseline_numpy = baseline.data.clone().cpu().numpy()
        progresses["baseline"].append(baseline_numpy)
        I_and_numpy = Iand.detach().cpu().numpy()
        progresses["I_and"].append(I_and_numpy)

        loss_info_str = f"[{it}/{niter}] loss: {total_loss_baseline:.4f}"
        pbar_iter.set_postfix_str(loss_info_str)
        if verbose == 1 or (it == niter - 1 or it % 10 == 0):
            print(loss_info_str)

    return baseline.detach(), losses, progresses


if __name__ == '__main__':
    print(get_reward2Iand_mat(4))