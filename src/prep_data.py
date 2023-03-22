import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

def prep_GNNB(task, bs):
    """
    description:
        get train and test loaders for a specific GNNBenchmark task
    
    params:
        task (str) : the task from GNNBenchmark [ZINC, COLLAB, CYCLES, MNIST, CIFAR, TSP]
        bs (int) : batch size
    """
    assert task in ["ZINC", "MNIST", "CIFAR", "TSP", "CYCLES", "COLLAB"], "Task not in collection. Pick from ZINC, COLLAB, CYCLES, MNIST, CIFAR, TSP."

    if task == "ZINC":
        train_set = ZINC("./data/", subset=True, split="train")
        test_set = ZINC("./data/", subset=True, split="test")
    else:
        train_set = GNNBenchmarkDataset("./data/", name=task, split="train")
        test_set = GNNBenchmarkDataset("./data/", name=task, split="test")

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)
    
    stats = {
        "indim": 1,
        "classes": 1
    }

    return stats, train_loader, test_loader