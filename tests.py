from main import main
from utils.argument import POSSIBLE_MODELS, POSSIBLE_TASKS
from pathlib import Path
import datetime
import logging
import traceback
import time

NUM_BATCHES = 10 #6
CHECKPOINT_INTERVAL = NUM_BATCHES // 2
REPORT_INTERVAL = NUM_BATCHES // 2
BATCH_SIZE = 2
LOG_LEVEL = logging.WARNING
SEED = int(time.time())

# For each mode, we iterate on different combinations of options to try.
# So, for each mode, we have a list of lists of options. Each list in the list is a combination of option to try.
MODEL_MODES = {"Rnn": [[],
                       ["-pgated=True"],
                       ["-pleaky=True"],
                       ], # the empty string is to also test the default
               "LSTM": [[],
                        ["-pchrono=True"],
                        ]}


folder = Path("test_logs")
folder.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
with open(folder/f"{timestamp}.log", "w") as f:
    for task in POSSIBLE_TASKS:
        for model in POSSIBLE_MODELS:
            for mode_options in MODEL_MODES[model]:
                try:
                    command = ["--task", task, "-pmodel_type=" + model,
                               f"-pnum_batches={NUM_BATCHES}", f"-pbatch_size={BATCH_SIZE}",
                               "--checkpoint_interval", str(CHECKPOINT_INTERVAL),
                               "--report_interval", str(REPORT_INTERVAL),
                               "--log_level", str(LOG_LEVEL),
                               "--seed", str(SEED)]
                    if mode_options:
                        command.extend(mode_options)

                    main(command)
                    message = f"SUCCESS - Task: {task + ',':10} Model: {model + ',':12}  Mode: {str(mode_options)}"
                    print(message)
                    f.write(message + "\n")
                except Exception as err:
                    message = (f"FAILED -  Task: {task + ',':10} Model: {model + ',':12}  Mode: {str(mode_options)},  "
                               #f"Error: \"{err.__class__.__name__}: {err}\""
                               f"\nStacktrace:\n\"\n{traceback.format_exc()}\"")
                    print(message + "\n")
                    f.write(message)