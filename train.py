from utils.Variable import Variable
from utils.reporting import get_ms, progress_bar, progress_clean
import numpy as np
import torch
import json
import logging
from pathlib import Path
from main import parse_param
from torch.nn import functional as F
LOGGER = logging.getLogger(__name__)


def train(task, args):

    num_batches = task.params.num_batches
    batch_size = task.params.batch_size

    LOGGER.info("Training model for %d batches (batch_size=%d)...",
                num_batches, batch_size)

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()
    scheduler = None
    if hasattr(task.__class__, "make_scheduler"):
        scheduler = task.__class__.make_scheduler(task.optimizer)

    for batch_num, x, y in task.dataloader:
        loss = train_batch(task.net, task.criterion, task.optimizer, x, y, task.__class__.loss_fn)
        if scheduler:
            scheduler.step(loss)
        losses += [loss]
        seq_lengths += [y.size(0)]

        # Update the progress bar
        progress_bar(batch_num, args.report_interval, loss)

        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            progress_clean()
            LOGGER.info("Batch %d Loss: %.6f Time: %d ms/sequence", batch_num, mean_loss, mean_time)
            start_ms = get_ms()

        # Checkpoint
        if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
            save_checkpoint(task.net, task.params.name, args,
                            batch_num, losses, costs, seq_lengths,
                            task)

    save_checkpoint(task.net, task.params.name, args, batch_num, losses, costs, seq_lengths, task)
    LOGGER.info("Done training.")


def train_batch(net, criterion, optimizer, X, Y, loss_fn):
    """Trains a single batch."""
    optimizer.zero_grad()
    loss = loss_fn(net, X, Y, criterion)
    loss.backward()
    optimizer.step()

    return loss.data[0]


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths, model):
    progress_clean()

    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"You need to create {checkpoint_path}"

    parsed = parse_param(args.param)
    if parsed.get("gated", False):
        modifier = "gated"
    elif parsed.get("leaky"):
        modifier = "leaky"
    elif parsed.get("chrono"):
        modifier = "chrono"
    else:
        modifier = "vanilla"

    basename = f"{checkpoint_path}/{name}-{net.__class__.__name__}-{modifier}-batch-{batch_num}-seed-{args.seed}"

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    history = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    with open(train_fname, 'wt') as f:
        f.write(json.dumps(history))

    model_fname = basename + ".pth"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)

    checkpoint = {
        "model": net.state_dict(),
        "history": history,
        # The two following things are required if we ever want to resume training
        "optimizer": model.optimizer,
        "batch_num": batch_num,
        "arguments": args,

    }
    torch.save(checkpoint, model_fname)

