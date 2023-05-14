import torch
import torch.nn as nn
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import time
import pandas as pd
import sys

sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
from tools.utils import plot_curves


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    device = next(model.parameters()).device

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator, desc="Training", mininterval=1):
        optimizer.zero_grad()

        text, text_lengths = batch.text
        text, text_lengths = text.to(device), text_lengths.to(device)
        label = batch.label.to(device)

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    device = next(model.parameters()).device

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            text, text_lengths = text.to(device), text_lengths.to(device)
            label = batch.label.to(device)

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_logistic_regression(args, model, train_iterator, test_iterator):
    criterion = nn.BCEWithLogitsLoss()
    if "model.pt" in os.listdir(args.save_root):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.save_root))
        model.load_state_dict(torch.load(os.path.join(args.save_root, "model.pt")))
        # evaluate the performance of the model
        eval_logistic_regression(model, test_iterator, criterion)
        return

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # set the decay of learning rate
    if args.logspace != 0:
        logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.n_epoch)
    else:
        logspace_lr = [args.lr] * args.n_epoch

    # define the train_csv
    learning_csv = os.path.join(args.save_root, "learning.csv")
    # define the res dict
    res_dict = {
        'train-loss': [], 'train-acc': [], 'test-loss': [], "test-acc": []
    }

    for epoch in range(args.n_epoch):

        if args.logspace != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]

        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

        # save the res in dict
        res_dict['train-loss'].append(train_loss)
        res_dict["train-acc"].append(train_acc)
        res_dict['test-loss'].append(test_loss)
        res_dict["test-acc"].append(test_acc)
        # store the res in csv
        pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

        plot_curves(args.save_root, res_dict)

    # save the model
    torch.save(model.cpu().state_dict(), os.path.join(args.save_root, "model.pt"))
    print("The model has been trained and saved in model path '{}'.".format(args.save_root))

    return model


def eval_logistic_regression(model, test_iterator, criterion):
    loss, acc = evaluate(model, test_iterator, criterion)
    print(f"[train_logistic_regression.py (eval_logistic_regression)] loss: {loss:.5f} | acc: {acc:.4f}")