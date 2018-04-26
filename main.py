import argparse
from task.taskManager import get_model
import train
import logging


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def parse_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='warpTask | addTask | copyTask')
    parser.add_argument('--model', required=True, help='Rnn | LRnn | GRnn')
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--report_interval', default=0.1, type=float, help="progress bar report interval")
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    init_logging()

    opt = parse_options()

    model = get_model(opt)
    # Train does not work yet
    train.train_model(model, opt)
