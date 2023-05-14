import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pprint import pprint
import doctest
from typing import Union, Iterable, List, Tuple, Callable
from tqdm import tqdm


def generate_all_masks(length: int) -> list:
    masks = list(range(2 ** length))
    masks = [np.binary_repr(mask, width=length) for mask in masks]
    masks = [[bool(int(item)) for item in mask] for mask in masks]
    return masks


def set_to_index(A):
    '''
    convert a boolean mask to an index
    :param A: <np.ndarray> bool (n_dim,)
    :return: an index

    [In] set_to_index(np.array([1, 0, 0, 1, 0]).astype(bool))
    [Out] 18
    '''
    assert len(A.shape) == 1
    A_ = A.astype(int)
    return np.sum([A_[-i-1] * (2 ** i) for i in range(A_.shape[0])])


def is_A_subset_B(A, B):
    '''
    Judge whether $A \subseteq B$ holds
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: Bool
    '''
    assert A.shape[0] == B.shape[0]
    return np.all(np.logical_or(np.logical_not(A), B))


def is_A_subset_Bs(A, Bs):
    '''
    Judge whether $A \subseteq B$ holds for each $B$ in 'Bs'
    :param A: <numpy.ndarray> bool (n_dim, )
    :param Bs: <numpy.ndarray> bool (n, n_dim)
    :return: Bool
    '''
    assert A.shape[0] == Bs.shape[1]
    is_subset = np.all(np.logical_or(np.logical_not(A), Bs), axis=1)
    return is_subset


def select_subset(As, B):
    '''
    Select A from As that satisfies $A \subseteq B$
    :param As: <numpy.ndarray> bool (n, n_dim)
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: a subset of As
    '''
    assert As.shape[1] == B.shape[0]
    is_subset = np.all(np.logical_or(np.logical_not(As), B), axis=1)
    return As[is_subset]


def set_minus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    calculate A/B
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: A\B

    >>> set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 0, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])

    >>> set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 1, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])
    '''
    assert A.shape[0] == B.shape[0] and len(A.shape) == 1 and len(B.shape) == 1
    A_ = A.copy()
    A_[B] = False
    return A_


def get_subset(A):
    '''
    Generate the subset of A
    :param A: <numpy.ndarray> bool (n_dim, )
    :return: subsets of A

    >>> get_subset(np.array([1, 0, 0, 1, 0, 1], dtype=bool))
    array([[False, False, False, False, False, False],
           [False, False, False, False, False,  True],
           [False, False, False,  True, False, False],
           [False, False, False,  True, False,  True],
           [ True, False, False, False, False, False],
           [ True, False, False, False, False,  True],
           [ True, False, False,  True, False, False],
           [ True, False, False,  True, False,  True]])
    '''
    assert len(A.shape) == 1
    n_dim = A.shape[0]
    n_subsets = 2 ** A.sum()
    subsets = np.zeros(shape=(n_subsets, n_dim)).astype(bool)
    subsets[:, A] = np.array(generate_all_masks(A.sum()))
    return subsets


def flatten(x):
    '''

    Flatten an irregular list of lists

    Reference <https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists>

    [In]  flatten(((1, 2), 3, 4)) -- Note: (with many brackets) x = ( (1, 2) , 3 , 4 )
    [Out] (1, 2, 3, 4)

    :param x:
    :return:
    '''
    if isinstance(x, Iterable):
        return list([a for i in x for a in flatten(i)])
    else:
        return [x]


def generate_subset_masks(set_mask, all_masks):
    '''
    For a given S, generate its subsets L's, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return: the subset masks, the bool indice
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = torch.logical_or(set_mask_, torch.logical_not(all_masks))
    is_subset = torch.all(is_subset, dim=1)
    return all_masks[is_subset], is_subset


def generate_reverse_subset_masks(set_mask, all_masks):
    '''
    For a given S, with subsets L's, generate N\L, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_rev_subset = torch.logical_or(set_mask_, all_masks)
    is_rev_subset = torch.all(is_rev_subset, dim=1)
    return all_masks[is_rev_subset], is_rev_subset


def generate_set_with_intersection_masks(set_mask, all_masks):
    '''
    For a given S, generate L's, s.t. L and S have intersection as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    have_intersection = torch.logical_and(set_mask_, all_masks)
    have_intersection = torch.any(have_intersection, dim=1)
    return all_masks[have_intersection], have_intersection


def get_reward(values, selected_dim, **kwargs):
    if selected_dim == "0":
        values = values[:, 0]
    elif selected_dim == "0-v0":
        assert "v0" in kwargs
        v0 = kwargs["v0"]
        values = values[:, 0] - v0
    elif selected_dim == "gt":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        values = values[:, gt]  # select the ground-truth dimension
    elif selected_dim == "gt-log-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        eps = 1e-7
        values = torch.softmax(values, dim=1)
        values = values[:, gt]
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "gt-log-odds-v0":
        assert "gt" in kwargs
        assert "v0" in kwargs
        gt = kwargs["gt"]
        v0 = kwargs["v0"]
        eps = 1e-7
        values = torch.softmax(values, dim=1)
        values = values[:, gt]
        values = torch.log(values / (1 - values + eps) + eps)
        values = values - v0
    elif selected_dim.startswith("gt-log-odds_t="):
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        temperature = float(selected_dim.split("=")[-1])
        eps = 1e-7
        values = torch.softmax(values / temperature, dim=1)
        values = values[:, gt]
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "max-log-odds":
        eps = 1e-7
        raise NotImplementedError  # TODO
        values = torch.softmax(values, dim=1)
        values = values[:, torch.argmax(values[-1])]
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "gt-logistic-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        assert gt == 0 or gt == 1
        assert len(values.shape) == 2 and values.shape[1] == 1
        if gt == 1:
            values = values[:, 0]
        else:
            values = -values[:, 0]
    elif selected_dim == "logistic-odds":
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = values[:, 0]
    elif selected_dim == "neg-logistic-odds":
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = -values[:, 0]
    elif selected_dim == "gt-logistic-odds-v0":
        assert "gt" in kwargs
        assert "v0" in kwargs
        gt = kwargs["gt"]
        v0 = kwargs["v0"]
        assert gt == 0 or gt == 1
        assert len(values.shape) == 2 and values.shape[1] == 1
        if gt == 1:
            values = values[:, 0]
        else:
            values = -values[:, 0]
        values = values - v0
    elif selected_dim == "gt-prob-log-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        assert gt == 0 or gt == 1
        eps = 1e-7
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = values[:, 0]
        if gt == 0:
            values = 1 - values
        else:
            values = values
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "prob-log-odds":
        eps = 1e-7
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = values[:, 0]
        if torch.round(values[-1]) == 0.:
            values = 1 - values
        else:
            values = values
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == None:
        values = values
    else:
        raise Exception(f"Unknown [selected_dim] {selected_dim}.")

    return values


def get_mask_input_func_image(grid_width: int) -> Callable:
    """
    Return the functional to mask an input image
    :param grid_width:
    :return:
    """
    def generate_masked_input(image: torch.Tensor, baseline: torch.Tensor, grid_indices_list: List):
        device = image.device
        _, image_channel, image_height, image_width = image.shape
        grid_num_h = int(np.ceil(image_height / grid_width))
        grid_num_w = int(np.ceil(image_width / grid_width))
        grid_num = grid_num_h * grid_num_w

        batch_size = len(grid_indices_list)
        masks = torch.zeros(batch_size, image_channel, grid_num)
        for i in range(batch_size):
            grid_indices = flatten(grid_indices_list[i])
            masks[i, :, list(grid_indices)] = 1

        masks = masks.view(masks.shape[0], image_channel, grid_num_h, grid_num_w)
        masks = F.interpolate(
            masks.clone(),
            size=[grid_width * grid_num_h, grid_width * grid_num_w],
            mode="nearest"
        ).float()
        masks = masks[:, :, :image_height, :image_width].to(device)

        expanded_image = image.expand(batch_size, image_channel, image_height, image_width).clone()
        expanded_baseline = baseline.expand(batch_size, image_channel, image_height, image_width).clone()
        masked_image = expanded_image * masks + expanded_baseline * (1 - masks)

        return masked_image

    return generate_masked_input


def get_mask_input_func_pointcloud(**kwargs) -> Callable:
    """
    Return the functional to mask an input point cloud
    :return:
    """
    def generate_masked_input(point_cloud: torch.Tensor, baseline: torch.Tensor, point_indices_list: List):
        device = point_cloud.device
        input_bs, c, n_points = point_cloud.shape
        assert input_bs == 1

        batch_size = len(point_indices_list)
        masks = torch.zeros(batch_size, c, n_points).to(device)
        for i in range(batch_size):
            point_indices = flatten(point_indices_list[i])
            masks[i, :, point_indices] = 1

        expanded_pc = point_cloud.expand(batch_size, c, n_points).clone()
        expanded_baseline = baseline.expand(batch_size, c, n_points).clone()
        masked_image = expanded_pc * masks + expanded_baseline * (1 - masks)

        return masked_image

    return generate_masked_input




def calculate_given_subset_outputs(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        player_masks: torch.BoolTensor,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None,
        verbose: int = 1
) -> (torch.Tensor, torch.Tensor):
    assert input.shape[0] == 1 and baseline.shape[0] == 1
    # ======================================================
    #     (1) First, generate the masked inputs
    # ======================================================
    if all_players is None:
        assert (background is None or len(background) == 0) and mask_input_fn is None
        masks = player_masks
        # masked_inputs = torch.where(masks, input.expand_as(masks), baseline.expand_as(masks))
    else:
        if background is None: background = []
        assert background is not None and mask_input_fn is not None
        all_players = np.array(all_players, dtype=object)
        grid_indices_list = []
        for i in range(player_masks.shape[0]):
            player_mask = player_masks[i].clone().cpu().numpy()
            grid_indices_list.append(list(flatten([background, all_players[player_mask]])))
        # masked_inputs = mask_input_fn(image=input, baseline=baseline, grid_indices_list=grid_indices_list)

    # ======================================================
    #  (2) Second, calculate the rewards of these inputs
    # ======================================================
    if calc_bs is None:
        calc_bs = player_masks.shape[0]

    outputs = []
    if verbose == 1:
        pbar = tqdm(range(int(np.ceil(player_masks.shape[0] / calc_bs))), ncols=100, desc="Calc model outputs")
    else:
        pbar = range(int(np.ceil(player_masks.shape[0] / calc_bs)))
    for batch_id in pbar:
        if all_players is None:
            masks_batch = masks[batch_id*calc_bs:(batch_id+1)*calc_bs]
            masked_inputs_batch = torch.where(masks_batch, input.expand_as(masks_batch), baseline.expand_as(masks_batch))
        else:
            grid_indices_batch = grid_indices_list[batch_id*calc_bs:(batch_id+1)*calc_bs]
            masked_inputs_batch = mask_input_fn(input, baseline, grid_indices_batch)
        output = model(masked_inputs_batch)
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)

    return player_masks, outputs



def calculate_all_subset_outputs(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None,
        verbose: int = 1
) -> (torch.Tensor, torch.Tensor):
    assert input.shape[0] == 1 and baseline.shape[0] == 1
    device = input.device
    if all_players is None:
        n_players = input.shape[1]
    else:
        n_players = len(all_players)
    player_masks = torch.BoolTensor(generate_all_masks(n_players)).to(device)
    return calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        player_masks=player_masks, all_players=all_players,
        background=background, mask_input_fn=mask_input_fn,
        calc_bs=calc_bs, verbose=verbose
    )


def calculate_output_N(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
        mask_input_fn: Callable = None,
        verbose: int = 1
):
    assert input.shape[0] == 1 and baseline.shape[0] == 1
    device = input.device
    if all_players is None:
        n_players = input.shape[1]
    else:
        n_players = len(all_players)
    player_masks = torch.ones(1, n_players).bool().to(device)
    _, output_N = calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None,
        player_masks=player_masks, verbose=verbose
    )
    return output_N


def calculate_output_empty(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
        mask_input_fn: Callable = None,
        verbose: int = 1
):
    assert input.shape[0] == 1 and baseline.shape[0] == 1
    device = input.device
    if all_players is None:
        n_players = input.shape[1]
    else:
        n_players = len(all_players)
    player_masks = torch.zeros(1, n_players).bool().to(device)
    _, output_empty = calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None,
        player_masks=player_masks, verbose=verbose
    )
    return output_empty



if __name__ == '__main__':
    # dim = 5
    # input = torch.randn(dim)
    # baseline = torch.FloatTensor([float(100 + 100 * i) for i in range(dim)])
    # model = nn.Linear(dim, 2)
    # calculate_all_subset_outputs_pytorch(model, input, baseline)

    # all_masks = generate_all_masks(6)
    # all_masks = torch.BoolTensor(all_masks)
    # set_mask = torch.BoolTensor([1, 0, 1, 1, 0, 0])
    # print(generate_subset_masks(set_mask, all_masks))

    # print(get_subset(np.array([1, 0, 0, 1, 0, 1]).astype(bool)))
    #
    # Bs = get_subset(np.array([1, 0, 0, 1, 0, 1]).astype(bool))
    # A = np.array([1, 0, 0, 1, 0, 0]).astype(bool)
    # print(is_A_subset_Bs(A, Bs))

    # all_masks = generate_all_masks(12)
    # all_masks = np.array(all_masks, dtype=bool)
    # set_index_list = []
    # for mask in all_masks:
    #     set_index_list.append(set_to_index(mask))
    # print(len(set_index_list), len(set(set_index_list)))
    # print(min(set_index_list), max(set_index_list))

    import doctest
    doctest.testmod()



    # S [1 0 0 1 0] subset(S) -> [4, 5]