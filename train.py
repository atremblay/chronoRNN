from utils.Variable import Variable
from utils.reporting import get_ms, progress_bar, progress_clean
import numpy as np
import torch
import json
import logging
from pathlib import Path
from torch.nn import functional as F
LOGGER = logging.getLogger(__name__)


def train_model(model, args):

    num_batches = model.params.num_batches
    batch_size = model.params.batch_size
    task = args.task

    LOGGER.info("Training model for %d batches (batch_size=%d)...",
                num_batches, batch_size)

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()

    for batch_num, x, y in model.dataloader:
        loss = train_batch(model.net, model.criterion, model.optimizer, x, y, task)
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
            save_checkpoint(model.net, model.params.name, args,
                            batch_num, losses, costs, seq_lengths)

    save_checkpoint(model.net, model.params.name, args, batch_num, losses, costs, seq_lengths)
    LOGGER.info("Done training.")


def train_batch(net, criterion, optimizer, X, Y, task):
    """Trains a single batch."""
    optimizer.zero_grad()
    inp_seq_len, batch_size, num_features = X.size()

    if task == 'warpTask' or task == "copyTask":
        loss = Variable(torch.zeros(1))
        net.create_new_state()
        for i in range(inp_seq_len):
            loss += criterion(net(X[i])[0], Y[i])

    elif task == "addTask":
        net.create_new_state()
        for i in range(inp_seq_len):
            output, hidden_state = net(X[i])
        loss = criterion(output, Y)

    else:
        raise RuntimeError("Unsupported task type")

    loss.backward()
    optimizer.step()

    return loss.data[0]


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths):
    progress_clean()

    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"You need to create {checkpoint_path}"
    #checkpoint_path.mkdir(exist_ok=True)

    basename = "{}/{}-{}-batch-{}".format(checkpoint_path, name, args.seed, batch_num)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))