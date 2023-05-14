import os
import os.path as osp
from tools.utils import makedirs, set_seed
import json


def init_dataset_model_settings(args):
    if args.dataset in ["census"] and args.arch in ["mlp2_logistic"]:
        args.model_kwargs = {"in_dim": 12, "hidd_dim": 100, "out_dim": 1}
        args.dataset_kwargs = {"balance": True}
        args.task = "logistic_regression"
    elif args.dataset in ["census"] and args.arch in ["mlp5", "resmlp5"]:
        args.model_kwargs = {"in_dim": 12, "hidd_dim": 100, "out_dim": 2}
        args.dataset_kwargs = {"balance": True}
        args.task = "classification"
    elif args.dataset in ["commercial"] and args.arch in ["mlp2_logistic"]:
        args.model_kwargs = {"in_dim": 10, "hidd_dim": 100, "out_dim": 1}
        args.dataset_kwargs = {"balance": True}
        args.task = "logistic_regression"
    elif args.dataset in ["commercial"] and args.arch in ["mlp5", "resmlp5"]:
        args.model_kwargs = {"in_dim": 10, "hidd_dim": 100, "out_dim": 2}
        args.dataset_kwargs = {"balance": True}
        args.task = "classification"
    elif args.dataset in ["bike"] and args.arch in ["mlp5", "resmlp5"]:
        args.model_kwargs = {"in_dim": 12, "hidd_dim": 100, "out_dim": 1}
        args.dataset_kwargs = {"balance": True}
        args.task = "regression"
    elif args.dataset in ["sst2", "cola"] and args.arch in ["lstm2_uni"]:
        args.model_kwargs = {"embedding_dim": 100, "hidden_dim": 256, "output_dim": 1}
        args.dataset_kwargs = {}
        args.task = "logistic_regression"
    elif args.dataset in ["sst2", "cola"] and args.arch in ["cnn"]:
        args.model_kwargs = {"embedding_dim": 100, "n_filters": 100, "output_dim": 1}
        args.dataset_kwargs = {"min_len": 5}
        args.task = "logistic_regression"
    else:
        raise NotImplementedError(f"[Undefined] Dataset: {args.dataset}, Model: {args.arch}")


def generate_dataset_model_desc(args):
    return f"dataset={args.dataset}" \
           f"_model={args.arch}" \
           f"_epoch={args.n_epoch}" \
           f"_bs={args.batch_size}" \
           f"_lr={args.lr}" \
           f"_logspace={args.logspace}" \
           f"_seed={args.seed}"


def makedirs_for_train_model(args):
    if args.dataset in ["cola", "sst2"]:
        if args.device >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    init_dataset_model_settings(args)
    args.dataset_model = generate_dataset_model_desc(args)

    set_seed(args.seed)

    args.save_root = osp.join(args.save_root, args.dataset_model)
    makedirs(args.save_root)

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
