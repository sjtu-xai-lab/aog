import os
import os.path as osp
import numpy as np
import pickle
import torch
import torch.nn as nn
import random
import torch.backends.cudnn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_seed(seed=0):
    """set the random seed for multiple packages.
    """
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    print(f"Set SEED: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # # Reference: <https://www.cnblogs.com/wanghui-garcia/p/11514502.html>
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # for ix, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = lr[0]
    return


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



def get_device(gpu_id):
    if gpu_id < 0:
        return torch.device("cpu")
    else:
        return torch.device(f"cuda:{gpu_id}")


def get_classification_accuracy(model, data_loader, device):
    if isinstance(model, nn.Module):
        model.eval()
    acc_avg = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"evaluating", mininterval=1, ncols=100)
        for images, labels in pbar:
            n_samples = images.shape[0]
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            y_pred = output.data.max(1)[1]
            acc_avg.update(y_pred.eq(labels.data).float().mean(), n_samples)
            pbar.set_postfix_str("acc={:.2f}%".format(acc_avg.avg * 100))
    print(f"Accuracy: {acc_avg.avg * 100}%")
    print()


def simplify_coalition_name(players: list):
    # players = long_name.split("-")
    # players = [int(player) for player in players]

    players = sorted(players)

    simplified_name = []

    seq = [players.pop(0)]
    while len(players) > 0:
        cur = players.pop(0)
        if cur == seq[-1] + 1:
            seq.append(cur)
        else:
            if len(seq) > 1:
                simplified_name.append(f"{seq[0]}to{seq[-1]}")
            else:
                simplified_name.append(f"{seq[0]}")
            seq = [cur]
    if len(seq) > 0:
        if len(seq) > 1:
            simplified_name.append(f"{seq[0]}to{seq[-1]}")
        else:
            simplified_name.append(f"{seq[0]}")
    return "-".join(simplified_name)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_curves(save_folder, res_dict):
    """plot curves for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
    """
    for key in res_dict.keys():
        # define the path
        path = os.path.join(save_folder, "{}-curve.png".format(key))
        # plot the fig
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(res_dict[key])) + 1, res_dict[key])
        ax.set(xlabel = 'epoch', ylabel = key,
            title = '{}\'s curve'.format(key))
        ax.grid()
        fig.savefig(path)
        plt.close()


if __name__ == '__main__':
    print(simplify_coalition_name("0-1-2-3-4-5-6-7-8-9"
                                  "-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-45-46-47-48-49-50-51-52-53-54-55-56-57-58-59-60-61-62-63-64-65-66-67-68-69-70-71-72-73-74-75-76-77-78-79-80-81-82-83-84-85-86-87-88-89-90-91-92-93-94-95-96-97-98-99-100-101-102-103-104-105-106-107-108-109-110-111-112-113-114-115-116-117-118-119-120-121-122-123-124-126-127-128-129-130-131-132-133-134-135-136-137-138-139-140-141-142-143-144-145-146-147-148-149-150-151-152-154-155-156-157-158-159-160-161-162-163-164-165-173-174-175-176-177-178-189"))
    print(simplify_coalition_name("1-3-4-5"))
