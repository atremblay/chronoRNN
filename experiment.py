import sys
import os
import train
from utils.argument import parse_options, init_logging, display_arguments
from task.taskManager import get_model
import evaluation
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
################################################################################
# Experiments
################################################################################


def uniform_warping_experiment():

    running_list = (#('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True', 5),
                    #('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -pgated=True -porthogonal_hidden_weight_init= -prmsprop_lr=2e-3', 5),
                    ('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -pleaky=True -porthogonal_hidden_weight_init= -prmsprop_lr=1e-3', 5),
                    )
    return _warping_experiment(running_list)


def warping_experiment():

    running_list = (#('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=', 5),
                    #('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp= -pgated=True -porthogonal_hidden_weight_init= -prmsprop_lr=2e-3', 5),
                    ('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp= -pleaky=True -porthogonal_hidden_weight_init= -prmsprop_lr=1e-3', 5),
                    )
    return _warping_experiment(running_list)


def uniform_padding_experiment():

    running_list = (#('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -ppadding_mode=True', 5),
                    ('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -pgated=True -ppadding_mode=True -porthogonal_hidden_weight_init= -prmsprop_lr=2e-3', 5),
                    #('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp=True -pleaky=True -ppadding_mode=True -porthogonal_hidden_weight_init= -prmsprop_lr=1e-3', 5),
                    )
    return _warping_experiment(running_list)


def padding_experiment():

    running_list = (#('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp= -ppadding_mode=True', 5),
                    ('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp= -pgated=True -ppadding_mode=True -porthogonal_hidden_weight_init= -prmsprop_lr=2e-3', 5),
                    #('--task warpTask --checkpoint_interval 0 -pmodel_type=Rnn -puniform_warp= -pleaky=True -ppadding_mode=True -porthogonal_hidden_weight_init= -prmsprop_lr=1e-3', 5),
                    )
    return _warping_experiment(running_list)


# Dictionary of experiments
exp = {'uniform_warping': uniform_warping_experiment,
       'warping':warping_experiment,
       'uniform_padding': uniform_padding_experiment,
       'padding':padding_experiment,
}

################################################################################
# Utilities for experiments
################################################################################

def _warping_experiment(running_list):

    for run in running_list:
        for max_warp in range(10, 51, 10):
            arguments = run[0] + ' -pmax_repeat=' + str(max_warp)
            for run_inst in range(run[1]):
                arguments_inst = arguments+' --run_instance ' + str(run_inst)
                yield arguments_inst.split(' ')


def evaluate_experiment(checkpoint_path):

    all_losses = {}
    for file in os.listdir(checkpoint_path):
        if '-leaky_RNN-' in file and file.endswith('.pth'):
            print(file)
            baseModel = file[:file.rfind('_')]
            model, forward_fn, loss_fn, dataloader_fn, history = evaluation.load_checkpoint(os.path.join(checkpoint_path, file), ('batch_size=1000', 'num_batches=10', 'epochs=1'))
            model.eval()
            this_loss = []

            for batch_num, x, y in dataloader_fn():
                losses = loss_fn(model, x, y)
                this_loss.append(losses.data.numpy().mean()/500) # Divide by sequence length

            print(this_loss)
            if baseModel.split('-')[2] in all_losses:
                if baseModel in all_losses[baseModel.split('-')[2]]:
                    all_losses[baseModel.split('-')[2]][baseModel][1].append(np.mean(this_loss).tolist())
                else:
                    all_losses[baseModel.split('-')[2]][baseModel] = [int(baseModel.split('-')[4]),[np.mean(this_loss).tolist()]]
            else:
                all_losses[baseModel.split('-')[2]] = {baseModel:[int(baseModel.split('-')[4]),[np.mean(this_loss).tolist()]]}

    pickle.dump(all_losses, open(os.path.join(checkpoint_path, 'eval.p'), "wb"))

    return all_losses

def plot_experiment(all_losses, checkpoint_path):

    # initial plot
    plots = []
    for model_architecture, losses in all_losses.items():
        # calculate the min and max series for each x
        x, min_ser, max_ser, avg_ser = [], [], [], []
        for x_i, loss in losses.values():
            min_ser.append(min(loss))
            max_ser.append(max(loss))
            avg_ser.append(np.mean(loss).tolist())
            x.append(x_i)

        Z = [(x_i, min_ser_i, max_ser_i, avg_ser_i) for x_i, min_ser_i, max_ser_i, avg_ser_i in
             sorted(zip(x, min_ser, max_ser, avg_ser))]
        x, min_ser, max_ser, avg_ser = zip(*Z)

        plot, = plt.plot(x, avg_ser, label=model_architecture)
        plots.append(plot)
        # plot the min and max series over the top
        plt.fill_between(x, min_ser, max_ser, alpha=0.2)

    plt.xlabel('Maximum warping')
    plt.ylabel('Loss after 3 epochs')
    plt.legend(handles=plots)
    plt.savefig(os.path.join(checkpoint_path, 'fig.png'))


def main(argv):
    opt = parse_options(argv)
    init_logging(opt.log_level)
    display_arguments(opt)
    model = get_model(opt)
    train.train(model, opt)


if __name__ == '__main__':
    random.seed(9001)
    checkpoint_path = './saves/'+sys.argv[1]
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

    checkpoint_path = ['--checkpoint_path', checkpoint_path]
    for run in exp[sys.argv[1]]():
        main(run+checkpoint_path)

    random.seed(5000)
    all_losses = evaluate_experiment(checkpoint_path[1])
    all_losses_1=pickle.load(open(os.path.join(checkpoint_path[1], 'eval_1.p'), "rb"))
    all_losses_1['leaky_RNN']=all_losses['leaky_RNN']
    plot_experiment(all_losses_1, checkpoint_path[1])
