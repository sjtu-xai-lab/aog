import os
import os.path as osp
import argparse
from datasets import get_dataset
from setup_exp import makedirs_for_train_model


def parse_args():
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument('--data-root', default='/data1/limingjie/data/tabular')
    parser.add_argument('--device', default=2, type=int)
    parser.add_argument("--dataset", default="census", type=str)
    parser.add_argument("--arch", default="mlp2_logistic", type=str)
    parser.add_argument("--save-root", default="../saved-models-tmp", type=str)

    parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
    parser.add_argument('--batch-size', default=512, type=int, help="set the batch size for training.")
    parser.add_argument('--lr', default=0.01, type=float, help="set the learning rate for training.")
    parser.add_argument("--logspace", default=1, type=int, help='the decay of learning rate.')
    parser.add_argument("--n-epoch", default=500, type=int, help='the number of iterations for training model.')

    args = parser.parse_args()
    makedirs_for_train_model(args)

    return args


def main_tabular(args):
    import models.tabular as models
    from tools.trainer.tabular import train_model

    print(args)
    print(f"dataset: {args.dataset} | arch: {args.arch}")

    # load dataset
    tabular = get_dataset(args.data_root, args.dataset)
    X_train, y_train, X_test, y_test = tabular.get_data()
    train_loader, test_loader = tabular.get_dataloader(batch_size=args.batch_size, **args.dataset_kwargs)
    X_train, y_train = X_train.to(args.device), y_train.to(args.device)
    X_test, y_test = X_test.to(args.device), y_test.to(args.device)

    # load model
    net = models.__dict__[args.arch](**args.model_kwargs)
    net = net.float().to(args.device)
    print(net)

    # train model
    train_model(args, net, X_train, y_train, X_test, y_test, train_loader, args.task)


def main_nlp(args):
    import models.nlp as models
    from tools.trainer.nlp import train_model

    print(args)
    print(f"dataset: {args.dataset} | arch: {args.arch}")

    # load dataset
    nlp_dataset = get_dataset(args.data_root, args.dataset, **args.dataset_kwargs)
    TEXT, LABEL = nlp_dataset.get_fields()
    train_loader, test_loader = nlp_dataset.get_dataloader(batch_size=args.batch_size)

    # load model
    net = models.__dict__[args.arch](
        vocab_size=len(TEXT.vocab),
        pad_idx=TEXT.vocab.stoi[TEXT.pad_token],
        **args.model_kwargs
    ).cuda()
    print(net)

    # train model
    net = train_model(args, net, train_loader, test_loader)


def main_image_tiny(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    if args.dataset in ["commercial", "census", "bike"]:
        main_tabular(args)
    elif args.dataset in ["sst2", "cola"]:
        main_nlp(args)
    elif args.dataset in ["simplemnist"]:
        raise NotImplementedError("Still under construction, stay tuned")
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")