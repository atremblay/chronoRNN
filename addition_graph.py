import torch
import json
from pathlib import Path
from task.taskManager import get_model
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from utils.Variable import maybe_cuda


def moving_average(values, window, stride):
    if window == 0:
        return values[::stride], np.arange(0, len(values), stride)
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma[::stride], np.arange(0, len(values), stride)


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: maybe_cuda(storage))
    task = get_model(checkpoint["arguments"])
    task.net.load_state_dict(checkpoint["model"])
    loss_fn = lambda model, x, y: task.__class__.loss_fn(model, x, y, task.criterion) # we hide the criterion
    forward_fn = task.__class__.forward_fn
    return task.net, forward_fn, loss_fn, task.dataloader_fn, checkpoint["history"], checkpoint["arguments"]


def parse_checkpoint_names(path):
    file_name = str(path).split("/")[-1]
    info = file_name.split("-")
    task = info[0]
    model = info[1]
    modifier = info[2]
    batch = int(info[4])
    seq_len = int(info[6])
    return task, model, modifier, batch, seq_len


max_y = {200: 0.25, 750: 0.225}
max_x = {200: 3000, 750: 30000}
window_size = {200: 1, 750: 1}
stride = {200: 20, 750: 60}
LINEWIDTH = 1.25
def main():
    ddd = defaultdict(lambda: defaultdict(dict))
    for path in Path("saves/").glob("*.pth"):
        task, model, modifier, batch, seq_len = parse_checkpoint_names(path)
        cp = load_checkpoint(path)
        history = cp[4]
        args = cp[5]
        ddd[seq_len][modifier][batch] = history

    for k0, v0 in ddd.items():
        for k1, v1 in v0.items():
            v0[k1] = max(v1.items(), key=lambda it: it[0])

    for k0, v0 in ddd.items():
        for k1, v1 in v0.items():
            print(f"{k0:4} {k1:>9} {v1[0]:7}")

    for k, v in ddd.items():
        print(f"{str(k) + ':':5} {list(v.keys())}")
        y0 = v["vanilla"][1]["loss"]
        y1 = v["chrono"][1]["loss"]
        y0_soft, x0_soft = moving_average(y0, window=window_size[k], stride=stride[k])
        y1_soft, x1_soft = moving_average(y1, window=window_size[k], stride=stride[k])
        plt.plot(np.linspace(0, max(x0_soft), len(x0_soft)), np.repeat(0.167, len(x0_soft)),
                 "--", label="Memoryless", linewidth=LINEWIDTH * 0.5, color="black")
        plt.plot(x0_soft, y0_soft, label="Vanilla", linewidth=LINEWIDTH)
        plt.plot(x1_soft, y1_soft, label=f"Chrono {k}", linewidth=LINEWIDTH)
        assert len(x0_soft) == len(x1_soft)
        plt.legend()
        plt.ylim(0, max_y[k])
        plt.xlim(0, max_x[k])
        plt.xlabel("Number of Iterations")
        plt.ylabel("MSE Loss")
        yticks = np.arange(6) * 0.05
        yticks = yticks[yticks <= max_y[k]]
        labels = yticks
        labels_str = [f"{np.floor(l * 100) / 100}" for l in labels]
        plt.yticks(labels, labels_str)
        plt.savefig(f"add_{k}.png", dpi=300, pad_inches=0, bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    main()