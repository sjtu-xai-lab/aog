from .train_logistic_regression import train_logistic_regression_model, eval_logistic_regression_model
from .train_classification import train_classification_model, eval_classification_model
from .train_regression import train_regression_model, eval_regression_model


def train_model(args, net, X_train, y_train, X_test, y_test, train_loader, task="classification"):
    if task == "logistic_regression":
        return train_logistic_regression_model(args, net, X_train, y_train, X_test, y_test, train_loader)
    elif task == "classification":
        return train_classification_model(args, net, X_train, y_train, X_test, y_test, train_loader)
    elif task == "regression":
        return train_regression_model(args, net, X_train, y_train, X_test, y_test, train_loader)
    else:
        raise NotImplementedError(f"Unknown task: {task}.")