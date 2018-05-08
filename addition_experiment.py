from main import main as train
from addition_graph import main as graph
import time
from pprint import pprint
import logging
LOGGER = logging.getLogger(__name__)


def stringify(dict_):
    for k, v in dict_.items():
        dict_[k] = str(v)
    return dict_


def compile_args(args):
    return [f"--{k}={v}" for k, v in args.items()]


def compile_pargs(args):
    return [f"-p{k}={v}" for k, v in args.items() if not v == "False"]


def compile(args, pargs_general, pargs_specific):
    args = compile_args(stringify(args))
    pargs_general = compile_pargs(stringify(pargs_general))
    pargs_specific = compile_pargs(stringify(pargs_specific))
    return args + pargs_general + pargs_specific


def main():
    seed = int(time.time() * 1000000)
    for chrono in [False, True]:
        args = dict(task="addTask", checkpoint_interval=3000,
                    seed=seed, log_level=logging.INFO)
        pargs_gen = dict(rmsprop_alpha=0.9,
                         rmsprop_momentum=0.9,
                         rmsprop_lr=1E-3,
                         batch_size=256,
                         model_type="LSTM",
                         chrono=chrono)
        pargs_short = dict(seq_len=200, num_batches=10000)
        pargs_long = dict(seq_len=750, num_batches=30000)
        compiled_short = compile(args, pargs_gen, pargs_short)
        compiled_long = compile(args, pargs_gen, pargs_long)
        pprint(compiled_short)
        train(compiled_short)
        pprint(compiled_long)
        train(compiled_long)

    graph()


if __name__ == "__main__":
    main()
