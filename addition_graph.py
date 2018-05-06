import torch
import json
from pathlib import Path
from task.taskManager import get_model
from collections import defaultdict
import matplotlib.pyplot as plt


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path)
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
    plt.plot(v["vanilla"][1]["loss"], label="vanilla")
    plt.plot(v["chrono"][1]["loss"], label="chrono")
    plt.legend()
    plt.ylim(0, 0.45)
    plt.show()
