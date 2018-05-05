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
    print(list(checkpoint.keys()))
    return task.net, forward_fn, loss_fn, task.dataloader_fn, checkpoint["history"], checkpoint["arguments"]


def parse_checkpoint_names(path):
    file_name = str(path).split("/")[-1]
    info = file_name.split("-")
    task = info[0]
    model = info[1]
    modifier = info[2]
    batch = int(info[4])
    return task, model, modifier, batch


ddd = defaultdict(dict)
for path in Path("saves/").glob("*.pth"):
    task, model, modifier, batch = parse_checkpoint_names(path)
    cp = load_checkpoint(path)
    history = cp[4]
    args = cp[5]
    ddd[batch][modifier] = history


for k, v in ddd.items():
    print(f"{str(k) + ':':10} {list(v.keys())}")
    vs = list(v.values())
    if len(vs) < 2:
        print("skipped")
        continue
    plt.plot(vs[0]["loss"])
    plt.plot(vs[1]["loss"])
    plt.ylim(0, 0.45)
    plt.show()
