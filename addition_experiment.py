from main import main
import logging

for bias in ["", "-pchrono=True "]:
    lr = 1E-3
    cmd = (f"--task=addTask --checkpoint_interval=1000 "
           f"-prmsprop_alpha=.9 -prmsprop_momentum=0 "
           f"-pseq_len=200 "
           f"-pnum_batches=6000 "
           f"-pbatch_size=32 "
           f"-pmodel_type=LSTM {bias}"
           f"-prmsprop_lr={lr} "
           f"--log_level {logging.INFO}"
           ).split(" ")
    print(cmd)
    main(cmd)

    cmd = (f"--task=addTask --checkpoint_interval=0 "
           f"-prmsprop_alpha=.9 -prmsprop_momentum=0 "
           f"-pseq_len=750 "
           f"-pnum_batches=20000 "
           f"-pbatch_size=32 "
           f"-pmodel_type=LSTM {bias}"
           f"-prmsprop_lr={lr}").split(" ")
    print(cmd)
    # main(cmd)
