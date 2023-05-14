from .tabular import TabularDataset
from .cola import CoLA
from .sst2 import SST2


def get_dataset(data_root, dataset_name, **kwargs):
    if dataset_name == "cola":
        return CoLA(data_root, **kwargs)
    elif dataset_name == "sst2":
        return SST2(data_root, **kwargs)
    elif dataset_name in ["census", "commercial", "bike"]:
        return TabularDataset(data_root, dataset_name)
    else:
        raise NotImplementedError