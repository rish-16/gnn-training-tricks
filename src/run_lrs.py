import torch
import torch.nn as nn
import torch_geometric as tg
import torch_geometric.nn as tgnn

import argparse
import numpy as np
import matplotlib.pyplot as plt

from prep_data import prep_GNNB
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--layer_type', type=str, default="GCN", choices=["GCN", "GIN", "GAT", "GPS"])
parser.add_argument('--task', type=str, default="ZINC", choices=["ZINC", "COLLAB", "MNIST", "CIFAR", "TSP", "CYCLES"])

args = parser.parse_args()

layer_type = args.layer_type
task = args.task

data_stats, train_loader, test_loader = prep_GNNB(task, bs=256)

def eval_model(model, loader, loss_fn):
    loss = 0
    n_samples = 0
    for batch in loader:
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        l = loss_fn(pred.view(-1), batch.y.view(-1))
        loss += l.cpu().detach().item()
        n_samples += 1

    return loss / n_samples

EPOCHS = 30
TRIALS = 3
all_train_losses = []
all_test_losses = []

for trial in range(TRIALS):
    model = build_gnn(layer_type)
    model = model(indim=data_stats["indim"], hidden=128, outdim=data_stats["classes"], n_layers=4)

    optimiser = torch.optim.AdamW(model.parameters())
    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    training_losses = []
    test_losses = []
    print (f"Trial: {trial}")

    for epoch in range(EPOCHS):
        epoch_loss = 0
        n_samples = 0
        for batch in train_loader:
            optimiser.zero_grad()
            batch = batch
            pred = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, ptr=batch.batch)
            loss = loss_fn(pred.view(-1), batch.y.view(-1))
            loss.backward()

            epoch_loss += loss.cpu().detach().item()
            n_samples += 1
            optimiser.step()
        # lr_schedule.step()

        epoch_loss = epoch_loss / n_samples
        test_loss = eval_model(model, test_loader, loss_fn)

        training_losses.append(epoch_loss)
        test_losses.append(test_loss)

        if epoch % 2 == 0:
            print (f"epoch: {epoch} | loss: {epoch_loss:.6f} | test loss: {test_loss:.6f}")

    all_train_losses.append(training_losses)
    all_test_losses.append(test_losses)

all_train_losses = np.array(all_train_losses)
all_test_losses = np.array(all_test_losses)

all_train_losses_mean = all_train_losses.mean(axis=0)
all_test_losses_mean = all_test_losses.mean(axis=0)

all_train_losses_std = all_train_losses.std(axis=0)
all_test_losses_std = all_test_losses.std(axis=0)

assert len(training_losses) == len(test_losses)

fig = plt.figure(figsize=(11, 7), dpi=80)
plt.plot(range(len(all_train_losses_mean)), all_train_losses_mean, label="train", color="blue", marker="*")
plt.fill_between(range(len(all_train_losses_mean)), all_train_losses_mean+all_train_losses_std, all_train_losses_mean-all_train_losses_std, alpha=0.4, facecolor="blue")

plt.plot(range(len(all_test_losses_mean)), all_test_losses_mean, label="test", color="orange", marker="o")
plt.fill_between(range(len(all_test_losses_mean)), all_test_losses_mean+all_test_losses_std, all_test_losses_mean-all_test_losses_std, alpha=0.4, facecolor="orange")

plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("MSE", fontsize=20)
# plt.title("AdamW + LR cosine annealing")
plt.title("Vanilla AdamW")
plt.grid(linestyle="dashed")
plt.legend()
plt.show()