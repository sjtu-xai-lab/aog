import os
import os.path as osp
from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle


# ==================================================
#               FOR  VISUALIZATION
# ==================================================

def plot_simple_line_chart(data, xlabel, ylabel, title, save_folder, save_name, X=None):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if X is None: X = np.arange(len(data))
    plt.plot(X, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def visualize_pattern_interaction(
        coalition_masks: np.ndarray,
        interactions: np.ndarray,
        attributes: List,
        std: np.ndarray = None,
        title: str = None,
        save_path="test.png"
):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 2)
    ax_attribute = plt.gca()
    x = np.arange(len(coalition_masks))
    y = np.arange(len(attributes))
    plt.xticks(x, [])
    plt.yticks(y, attributes)
    plt.xlim(x.min() - 0.5, x.max() + 0.5)
    plt.ylim(y.min() - 0.5, y.max() + 0.5)
    plt.xlabel(r"interaction pattern $S$")
    plt.ylabel("attribute")

    patch_colors = {
        True: {
            'pos': 'red',
            'neg': 'blue'
        },
        False: 'gray'
    }
    patch_width = 0.8
    patch_height = 0.9

    for coalition_id in range(len(coalition_masks)):
        coalition = coalition_masks[coalition_id]
        for attribute_id in range(len(attributes)):
            # is_selected = judge_is_selected(attributes[attribute_id], attribute_id, coalition)
            is_selected = coalition[attribute_id]
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if interactions[coalition_id] > 0: facecolor = patch_colors[is_selected]['pos']
                else: facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(coalition_id - patch_width / 2,
                    attribute_id - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax_attribute.add_patch(rect)

    plt.subplot(2, 1, 1, sharex=ax_attribute)
    if title is not None:
        plt.title(title)
    plt.ylabel(r"interaction strength $|I(S)|$")
    # plt.yscale("log")
    ax_eval = plt.gca()
    plt.setp(ax_eval.get_xticklabels(), visible=False)
    ax_eval.spines['right'].set_visible(False)
    ax_eval.spines['top'].set_visible(False)
    plt.plot(np.arange(len(coalition_masks)), np.abs(interactions))
    if std is not None:
        plt.fill_between(
            np.arange(len(coalition_masks)),
            np.abs(interactions) - std,
            np.abs(interactions) + std, alpha=0.5
        )
    plt.hlines(y=0, xmin=0, xmax=len(coalition_masks), linestyles='dotted', colors='red')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def generate_colorbar(ax, cmap_name, x_range, loc, title=""):
    '''
    generate a (fake) colorbar in a matplotlib plot
    :param ax:
    :param cmap_name:
    :param x_range:
    :param loc:
    :param title:
    :return:
    '''
    length = x_range[1] - x_range[0] + 1
    bar_ax = ax.inset_axes(loc)
    bar_ax.set_title(title)
    dummy = np.vstack([np.linspace(0, 1, length)] * 2)
    bar_ax.imshow(dummy, aspect='auto', cmap=plt.get_cmap(cmap_name))
    bar_ax.set_yticks([])
    bar_ax.set_xticks(x_range)


def plot_interaction_progress(interaction, save_path, order_cfg="descending", title=""):
    if not isinstance(interaction, list):
        interaction = [interaction]

    order_first = np.argsort(-interaction[0])

    plt.figure(figsize=(8, 6))
    plt.title(title)

    cmap_name = 'viridis'
    colors = cm.get_cmap(name=cmap_name, lut=len(interaction))
    colors = colors(np.arange(len(interaction)))

    label = None
    for i, item in enumerate(interaction):
        X = np.arange(1, item.shape[0] + 1)
        plt.hlines(0, 0, X.shape[0], linestyles="dotted", colors="red")
        label = f"iter {i+1}" if len(interaction) > 1 else None
        if order_cfg == "descending":
            plt.plot(X, item[np.argsort(-item)], label=label, color=colors[i])
        elif order_cfg == "first":
            plt.plot(X, item[order_first], label=label, color=colors[i])
        else:
            raise NotImplementedError(f"Unrecognized order configuration {order_cfg}.")
        plt.xlabel("patterns (with I(S) descending)")
        plt.ylabel("I(S)")
    # if label is not None: plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    generate_colorbar(
        ax, cmap_name,
        x_range=(0, len(interaction) - 1),
        loc=[0.58, 0.9, 0.4, 0.03],
        title="checkpoint id"
    )
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def plot_interaction_strength_ratio(interaction, save_path, title="Relationship between # patterns & explain-ratio"):
    strength = np.abs(interaction)
    strength = strength[np.argsort(-strength)]
    total_strength = strength.sum()
    strength = strength / total_strength
    plt.figure()
    cum_strength = np.cumsum(strength)
    plt.plot(np.arange(len(interaction)), cum_strength)

    for thres in [0.7, 0.8, 0.9, 0.95]:
        plt.hlines(y=thres, xmin=0, xmax=len(interaction)-1, linestyles="dashed", colors="red")
        idx = np.where(cum_strength >= thres)[0][0]
        plt.scatter(idx, cum_strength[idx], c="red")
        plt.annotate(f"{idx}", (idx, cum_strength[idx]), zorder=5)

    plt.title(title)
    plt.xlabel(r"# of patterns $S$")
    plt.ylabel(r"ratio")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, transparent=True)
    plt.close("all")


def plot_interaction_strength_descending(interaction, save_path, title="interaction strength (descending)", standard=None):
    strength = np.abs(interaction)
    strength = strength[np.argsort(-strength)]

    plt.figure()
    plt.plot(np.arange(len(interaction)), strength)
    if standard is not None:
        for r in [1.0, 0.1, 0.05, 0.01]:
            plt.hlines(y=r*standard, xmin=0, xmax=len(interaction)-1, linestyles="dashed", colors="red")
            idx = np.where(strength <= r*standard)[0][0]
            plt.scatter(idx, strength[idx], c="red")
            plt.annotate(f"{idx}", (idx, strength[idx]), zorder=5)
    plt.title(title)
    plt.xlabel(r"# of patterns $S$")
    plt.ylabel(r"$|I(S)|$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, transparent=True)
    plt.close("all")


def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = np.array(mean).reshape((-1, 1, 1))
    std = np.array(std).reshape((-1, 1, 1))
    image = image * std + mean
    return image


def plot_image(image, save_folder, save_name):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(image.transpose(1, 2, 0).clip(0, 1))
    plt.axis("off")
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), transparent=True)
    plt.close("all")
