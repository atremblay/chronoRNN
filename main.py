import argparse
from task.taskManager import get_model
import train
import logging
LOGGER = logging.getLogger(__name__)

def init_logging(level):
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=level)


def display_arguments(opt):
    opt_dict = vars(opt)
    formatted_opt = [f"\t- {k + ' :':23} {v}" for k, v in opt_dict.items()]
    joined_opt = "\n".join(formatted_opt)
    LOGGER.debug(f"\nCommand-line arguments:\n{joined_opt}")


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='warpTask | addTask | copyTask')
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--report_interval', default=1000, type=int, help="progress bar report interval")
    parser.add_argument("--checkpoint_interval", default=1000, type=int, help="save a checkpoint every x batches")
    parser.add_argument("--checkpoint_path", default="./saves", )
    parser.add_argument("--seed", type=int, default=0, )
    parser.add_argument("--log_level", type=int, default=logging.DEBUG,)
    opt = parser.parse_args()

    assert opt.log_level in {0, 10, 20, 30, 40, 50}, (
        "\n\nBad log_level argument. Valid log levels are:\n"
        "\t- CRITICAL   50\n"
        "\t- ERROR      40\n"
        "\t- WARNING    30\n"
        "\t- INFO       20\n"
        "\t- DEBUG      10\n"
        "\t- NOTSET     0\n"
        f"Got '{opt.log_level}', of the type '{type(opt.log_level)}'.")

    return opt


if __name__ == '__main__':
    opt = parse_options()
    init_logging(opt.log_level)
    display_arguments(opt)
    model = get_model(opt)
    # Train does not work yet
    train.train_model(model, opt)
