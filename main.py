import sys
import argparse
import train
import logging
from pathlib import Path
from task.taskManager import get_model

POSSIBLE_TASKS = {"warpTask", "addTask", "copyTask"}
POSSIBLE_MODELS = {"ChronoLSTM", "Rnn"}

LOGGER = logging.getLogger(__name__)


def init_logging(level):
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=level)


def display_arguments(opt):
    opt_dict = vars(opt)
    formatted_opt = [f"\t- {k + ' :':23} {v}" for k, v in opt_dict.items()]
    joined_opt = "\n".join(formatted_opt)
    LOGGER.debug(f"\nCommand-line arguments:\n{joined_opt}\n")


def parse_param(param):
    splitted = (x.split("=") for x in param)
    return {y[0]: y[1] for y in splitted}


def validate_arguments(opt):
    param = parse_param(opt.param)
    if "model_type" in param:
        assert param["model_type"] in POSSIBLE_MODELS, (f"Got -pmodel_type '{param['model_type']}', " 
                                                        f"expected one of {POSSIBLE_MODELS}")

    assert opt.task in POSSIBLE_TASKS, f"Got --task '{opt.task}', expected one of {POSSIBLE_TASKS}"
    assert opt.report_interval > 0, (f"opt.report_interval needs to be a non-zero positive int," 
                                     f" got '{opt.checkpoint_interval}'")
    assert opt.checkpoint_interval >= 0, (f"opt.checkpoint_interval needs to be a positive int," 
                                          f" got '{opt.checkpoint_interval}'")
    assert opt.checkpoint_path.exists(), (f"--checkpoint_path '{opt.checkpoint_path}' does not exist. "
                                          f"You need to create it.")
    assert opt.log_level in {0, 10, 20, 30, 40, 50}, (
        "\n\nBad log_level argument. Valid log levels are:\n"
        "\t- CRITICAL   50\n"
        "\t- ERROR      40\n"
        "\t- WARNING    30\n"
        "\t- INFO       20\n"
        "\t- DEBUG      10\n"
        "\t- NOTSET     0\n"
        f"Got '{opt.log_level}', of the type '{type(opt.log_level)}'.")


def parse_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help=" | ".join(POSSIBLE_TASKS))
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--report_interval', default=1000, type=int, help="progress bar report interval")
    parser.add_argument("--checkpoint_interval", default=1000, type=int,
                        help="Save a checkpoint every x batches. 0 means only at the end of the training")
    parser.add_argument("--checkpoint_path", default="./saves", type=Path)
    parser.add_argument("--seed", type=int, default=0, )
    parser.add_argument("--log_level", type=int, default=logging.DEBUG,)
    opt = parser.parse_args(argv)
    validate_arguments(opt)
    return opt


def main(argv):
    opt = parse_options(argv)
    init_logging(opt.log_level)
    display_arguments(opt)
    model = get_model(opt)
    # Train does not work yet
    train.train(model, opt)


if __name__ == '__main__':
    main(sys.argv[1:])