import torch
from pathlib import Path
from task.taskManager import get_model

def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path)
    task = get_model(checkpoint["arguments"])
    task.net.load_state_dict(checkpoint["model"])
    loss_fn = lambda model, x, y: task.__class__.loss_fn(model, x, y, task.criterion) # we hide the criterion
    forward_fn = task.__class__.forward_fn
    return task.net, forward_fn, loss_fn, task.dataloader_fn, checkpoint["history"]
