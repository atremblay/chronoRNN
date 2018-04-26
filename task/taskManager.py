import logging
import re
import sys
import attr

LOGGER = logging.getLogger(__name__)


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
