import sys
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
import models.tabular as model
from tools.utils import plot_curves


def train_classification_model(args, net, X_train, y_train, X_test, y_test, train_loader):
    """prepare the model the later usage. If the model haven't been trained, we will firstly
    train the model. If the model has been trained, we will certainly load the model.

    Args:
        args (Namespace): set containing all program arguments.
        X_train, y_train, X_test, y_test (tensor.to(deivice)): splitted dataset.
        train_loader (nn.dataloader) used for training model

    Returns:
        return the fitted model or trained model.
    """

    # define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)

    if "model.pt" in os.listdir(args.save_root):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.save_root))
        net.load_state_dict(torch.load(os.path.join(args.save_root, "model.pt")))

        # evaluate the performance of the model
        net = eval_classification_model(net, X_test, y_test, criterion)

        return net
    else:
        print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.save_root))
        if args.model_seed != args.seed:
            print("argument model_seed should be as same as argument seed to train a new model with new settings. Terminate program with code -3.")
            exit(-3)

        # define the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        # set the decay of learning rate
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.n_epoch)

        # define the train_csv
        learning_csv = os.path.join(args.save_root, "learning.csv")
        # define the res dict
        res_dict = {
            'train-loss': [], 'train-acc': [], 'test-loss': [], "test-acc": []
        }

        if args.ckpt_every is not None:
            os.makedirs(os.path.join(args.save_root, "checkpoints"), exist_ok=True)
            torch.save(net.state_dict(), os.path.join(args.save_root,  "checkpoints", "model_init.pt"))

        # starts training
        for epoch in range(args.n_epoch + 1):  # to test the acc for the last time
            # eval on test set
            net.eval()
            with torch.no_grad():
                # eval on train
                train_log_probs = net(X_train)
                train_loss = criterion(train_log_probs, y_train)
                train_targets = torch.argmax(train_log_probs, dim=-1)
                train_acc = torch.sum(train_targets == y_train).item() / len(y_train)

                # eval on test
                test_log_probs = net(X_test)
                test_loss = criterion(test_log_probs, y_test)
                test_targets = torch.argmax(test_log_probs, dim=-1)
                test_acc = torch.sum(test_targets == y_test).item() / len(y_test)

            # save the res in dict
            res_dict['train-loss'].append(train_loss.item())
            res_dict["train-acc"].append(train_acc)
            res_dict['test-loss'].append(test_loss.item())
            res_dict["test-acc"].append(test_acc)
            # store the res in csv
            pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

            # show loss
            if epoch % 10 == 0 or epoch == args.n_epoch:
                print('On train set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, train_loss.item(), train_acc))
                print('On test set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, test_loss.item(), test_acc))

                if epoch % 100 == 0 or epoch == args.n_epoch - 1:
                    # draw the curves
                    plot_curves(args.save_root, res_dict)

            if epoch >= args.n_epoch:
                break

            # set the lr
            if args.logspace != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]
            for step, (batch_X, batch_y) in enumerate(train_loader):
                # print(batch_y.sum() / len(batch_y))
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                net.train()
                net.zero_grad()
                log_probs = net(batch_X)
                loss = criterion(log_probs, batch_y)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                loss.backward()
                optimizer.step()

            if args.ckpt_every is not None and (epoch + 1) % args.ckpt_every == 0:
                torch.save(net.state_dict(), os.path.join(args.save_root, "checkpoints", f"model_{epoch+1}.pt"))

        plot_curves(args.save_root, res_dict)
        # save the model
        torch.save(net.cpu().state_dict(), os.path.join(args.save_root, "model.pt"))
        print("The model has been trained and saved in model path '{}'.".format(args.save_root))

        return net


def eval_classification_model(net, X_test, y_test, criterion):
    # evaluate the performance of the model
    net.eval()
    with torch.no_grad():
        logits = net(X_test)
        loss = criterion(logits, y_test)
        targets = torch.argmax(logits, dim=-1)
        acc = torch.sum(targets == y_test).item() / len(y_test)

    print('On test set - \t Loss: {:.6f} \t Acc: {:.9f}'.format(loss.item(), acc))

    return net
