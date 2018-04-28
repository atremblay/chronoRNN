import sys
import train
from utils.argument import parse_options, init_logging, display_arguments
from task.taskManager import get_model


def main(argv):
    opt = parse_options(argv)
    init_logging(opt.log_level)
    display_arguments(opt)
    model = get_model(opt)
    train.train(model, opt)


if __name__ == '__main__':
    main(sys.argv[1:])