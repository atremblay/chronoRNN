import sys
import train
from utils.argument import parse_options, init_logging, display_arguments
from task.taskManager import get_model


def uniform_warping_experiment():

    running_list = (('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True', 5),
                    ('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -pgated=True', 5),
                    ('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -pleaky=True', 5),
                    )


    for run in running_list:
        for seq_len in range(10,101):
            arguments = run[0] + ' -pseq_len=' + str(seq_len)
            for run_inst in range(run[1]):
                arguments_inst = arguments+' --run_instance ' + str(run_inst)
                yield arguments_inst.split(' ')


def main(argv):
    opt = parse_options(argv)
    init_logging(opt.log_level)
    display_arguments(opt)
    model = get_model(opt)
    train.train(model, opt)


# Dictionary of experiments
exp = {'uniform_warping': uniform_warping_experiment,
}

if __name__ == '__main__':
    for run in exp[sys.argv[1]]():
        main(run)