import torch
import torch.nn as nn
from .train_logistic_regression import train_logistic_regression, eval_logistic_regression


def train_model(args, model, train_iterator, test_iterator):
    if args.task == "logistic_regression":
        return train_logistic_regression(args, model, train_iterator, test_iterator)


def eval_model(args, model, train_iterator, test_iterator):
    if args.task == "logistic_regression":
        print("[train.py (eval_model)] On train set:")
        eval_logistic_regression(model, train_iterator, nn.BCEWithLogitsLoss())
        print("[train.py (eval_model)] On test set:")
        eval_logistic_regression(model, test_iterator, nn.BCEWithLogitsLoss())
