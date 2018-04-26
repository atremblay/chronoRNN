import logging
import re
import sys
import attr
from utils.Variable import Variable
import torch
LOGGER = logging.getLogger(__name__)


def per_sequence_item_loss(net, X, Y, criterion):
    loss = Variable(torch.zeros(1))
    net.create_new_state()
    for i in range(X.size(0)):
        loss += criterion(net(X[i])[0], Y[i])
    return loss


def per_full_sequence_loss(net, X, Y, criterion):
    net.create_new_state()
    for i in range(X.size(0)):
        output, hidden_state = net(X[i])
    loss = criterion(output, Y)
    return loss


def per_sequence_item_forward(net, X,):
    net.create_new_state()
    outputs = [] # The outputs can be of a variable size.
    for i in range(X.size(0)):
        outputs.append(net(X[i])[0])
    return outputs


def per_full_sequence_forward(net, X):
    net.create_new_state()
    output = None
    for i in range(X.size(0)):
        output = net(X[i])[0]
    return output


def get_model(opt):

    LOGGER.info("Training for the **%s** task", opt.task)
    mods = getattr(__import__('task.' + opt.task), opt.task)

    model_cls = mods.TaskModelTraining
    params_cls = mods.TaskParams
    params = params_cls()
    params = update_model_params(params, opt.param)

    model = model_cls(params=params)

    LOGGER.info(params)

    return model


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params
