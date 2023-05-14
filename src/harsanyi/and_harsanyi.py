import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from typing import Union, Iterable, List, Tuple, Callable

from .and_harsanyi_utils import get_reward2Iand_mat, _train_baseline
from .interaction_utils import calculate_all_subset_outputs, calculate_output_empty, calculate_output_N, get_reward, calculate_given_subset_outputs
from .plot import plot_simple_line_chart, plot_interaction_progress


class AndHarsanyi(object):
    def __init__(
            self,
            model: Union[nn.Module, Callable],
            selected_dim: Union[None, str],
            x: torch.Tensor,
            baseline: torch.Tensor,
            y: Union[torch.Tensor, int, None],
            all_players: Union[None, tuple, list] = None,
            background: Union[None, tuple, list] = None,
            mask_input_fn: Callable = None,
            calc_bs: int = None,
            verbose: int = 1,
    ):
        assert x.shape[0] == baseline.shape[0] == 1
        self.model = model
        self.selected_dim = selected_dim
        self.input = x
        self.target = y
        self.baseline = baseline
        self.device = x.device
        self.verbose = verbose

        self.all_players = all_players  # customize players
        if background is None:
            background = []
        self.background = background  # players that always exists (default: emptyset [])

        self.mask_input_fn = mask_input_fn  # for image data
        self.calc_bs = calc_bs

        if all_players is not None:  # image data
            self.n_players = len(all_players)
        else:
            self.n_players = self.input.shape[1]

        if self.verbose == 1:
            print("[AndHarsanyi] Generating v->I^and matrix:")
        self.reward2Iand = get_reward2Iand_mat(self.n_players).to(self.device)
        if self.verbose == 1:
            print("[AndHarsanyi] Finish.")

        # calculate v(N) and v(empty)
        with torch.no_grad():
            self.output_empty = calculate_output_empty(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, verbose=self.verbose
            )
            self.output_N = calculate_output_N(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, verbose=self.verbose
            )
            # self.output_empty = model(self.baseline)
            # self.output_N = model(self.input)
        if self.selected_dim.endswith("-v0"):
            self.v0 = get_reward(self.output_empty, self.selected_dim[:-3], gt=y)
        else:
            self.v0 = 0
        self.v_N = get_reward(self.output_N, self.selected_dim, gt=y, v0=self.v0)
        self.v_empty = get_reward(self.output_empty, self.selected_dim, gt=y, v0=self.v0)

    def attribute(self):
        with torch.no_grad():
            self.masks, outputs = calculate_all_subset_outputs(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs,
                verbose=self.verbose
            )
        self.rewards = get_reward(outputs, self.selected_dim, gt=self.target, v0=self.v0)
        self.Iand = torch.matmul(self.reward2Iand, self.rewards)

    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        masks = self.masks.cpu().numpy()
        rewards = self.rewards.cpu().numpy()
        sample = self.input.cpu().numpy()
        np.save(osp.join(save_folder, "rewards.npy"), rewards)
        np.save(osp.join(save_folder, "masks.npy"), masks)
        np.save(osp.join(save_folder, "sample.npy"), sample)
        Iand = self.Iand.cpu().numpy()
        np.save(osp.join(save_folder, "Iand.npy"), Iand)

    def get_interaction(self):
        return self.Iand

    def get_masks(self):
        return self.masks

    def get_rewards(self):
        return self.rewards



class AndBaselineSparsifier(object):
    def __init__(
            self,
            calculator: AndHarsanyi,
            loss: str,
            baseline_min: Union[float, torch.Tensor],
            baseline_max: Union[float, torch.Tensor],
            baseline_lr: float,
            niter: int,
    ):
        self.calculator = calculator

        # hyper-parameters for learning baseline values
        self.baseline_lr = baseline_lr
        self.baseline_min = baseline_min
        self.baseline_max = baseline_max

        # general hyper-parameters
        self.niter = niter
        assert loss in ["l1"]
        self.loss = loss

        # initialize the baseline values
        self.baseline_init = calculator.baseline.clone()
        self.baseline = calculator.baseline.clone()

    def sparsify(self, verbose_folder=None):
        def f_baseline2rewards(baseline):
            masks, outputs = calculate_all_subset_outputs(
                model=self.calculator.model, input=self.calculator.input, baseline=baseline,
                all_players=self.calculator.all_players, background=self.calculator.background,
                mask_input_fn=self.calculator.mask_input_fn, calc_bs=self.calculator.calc_bs,
                verbose=self.calculator.verbose
            )
            rewards = get_reward(outputs, self.calculator.selected_dim, gt=self.calculator.target)
            return masks, rewards

        def f_masksbaseline2rewards(masks, baseline):
            masks, outputs = calculate_given_subset_outputs(
                model=self.calculator.model, input=self.calculator.input, baseline=baseline,
                all_players=self.calculator.all_players, background=self.calculator.background,
                player_masks=masks, mask_input_fn=self.calculator.mask_input_fn,
                calc_bs=self.calculator.calc_bs, verbose=self.calculator.verbose
            )
            rewards = get_reward(outputs, self.calculator.selected_dim, gt=self.calculator.target)
            return masks, rewards

        baseline, losses, progresses = _train_baseline(
            f_baseline2rewards=f_baseline2rewards,
            f_masksbaseline2rewards=f_masksbaseline2rewards,
            baseline_init=self.baseline,
            loss_type=self.loss,
            baseline_lr=self.baseline_lr,
            niter=self.niter,
            baseline_min=self.baseline_min,
            baseline_max=self.baseline_max,
            reward2Iand=self.calculator.reward2Iand,
            calc_bs=self.calculator.calc_bs,
            verbose=self.calculator.verbose
        )
        self.baseline = baseline.clone()

        self._calculate_interaction()

        if verbose_folder is None:
            return

        for k in losses.keys():
            plot_simple_line_chart(
                data=losses[k], xlabel="iteration", ylabel=f"{k}", title="",
                save_folder=verbose_folder, save_name=f"{k}_curve_optimize_p_q"
            )
        for k in progresses.keys():
            plot_interaction_progress(
                interaction=progresses[k], save_path=osp.join(verbose_folder, f"{k}_progress_optimize_p_q.png"),
                order_cfg="descending", title=f"{k} progress during optimization"
            )


        with open(osp.join(verbose_folder, "log.txt"), "w") as f:
            f.write(f"loss: {self.loss} | baseline_lr: {self.baseline_lr} | niter: {self.niter}\n")
            f.write(f"\tSum of I^and: {torch.sum(self.Iand)}\n")
            f.write(f"\t|I^and|_1: {torch.sum(torch.abs(self.Iand)).item()}\n")
            f.write("\tAfter optimization,\n")
            for k, v in losses.items():
                f.write(f"\t\t{k}: {v[0]} -> {torch.sum(torch.abs(self.Iand)).item()}\n")

    def _calculate_interaction(self):
        with torch.no_grad():
            _, outputs = calculate_all_subset_outputs(
                model=self.calculator.model, input=self.calculator.input, baseline=self.baseline,
                all_players=self.calculator.all_players, background=self.calculator.background,
                mask_input_fn=self.calculator.mask_input_fn, calc_bs=self.calculator.calc_bs,
                verbose=self.calculator.verbose
            )
        self.rewards = get_reward(outputs, self.calculator.selected_dim, gt=self.calculator.target)
        self.Iand = torch.matmul(self.calculator.reward2Iand, self.rewards).detach()

    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        self._calculate_interaction()
        masks = self.calculator.masks.cpu().numpy()
        baseline = self.baseline.detach().cpu().numpy()
        rewards = self.rewards.cpu().numpy()
        Iand = self.Iand.cpu().numpy()
        np.save(osp.join(save_folder, "masks.npy"), masks)
        np.save(osp.join(save_folder, "baseline.npy"), baseline)
        np.save(osp.join(save_folder, "rewards.npy"), rewards)
        np.save(osp.join(save_folder, "Iand.npy"), Iand)

    def get_interaction(self):
        return self.Iand

    def get_baseline(self):
        return self.baseline

    def get_masks(self):
        return self.calculator.get_masks()

    def get_rewards(self):
        return self.rewards

