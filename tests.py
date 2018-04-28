from main import main, POSSIBLE_MODELS, POSSIBLE_TASKS
from pathlib import Path
import datetime
import logging
import traceback

NUM_BATCHES = 6
CHECKPOINT_INTERVAL = NUM_BATCHES // 2
REPORT_INTERVAL = NUM_BATCHES // 2
BATCH_SIZE = 2
LOG_LEVEL = logging.WARNING

folder = Path("test_logs")
folder.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
with open(folder/f"{timestamp}.log", "w") as f:
    for task in POSSIBLE_TASKS:
        for model in POSSIBLE_MODELS:
            try:
                main(["--task", task, "-pmodel_type=" + model,
                      f"-pnum_batches={NUM_BATCHES}", f"-pbatch_size={BATCH_SIZE}",
                      "--checkpoint_interval", str(CHECKPOINT_INTERVAL),
                      "--report_interval", str(REPORT_INTERVAL),
                      "--log_level", str(LOG_LEVEL)],)
                message = f"SUCCESS - Task: {task + ',':10} Model: {model}"
                print(message)
                f.write(message + "\n")
            except Exception as err:
                message = (f"FAILED -  Task: {task + ',':10} Model: {model},  Error: \"{err.__class__.__name__}: {err}\""
                           f"\nStacktrace:\n\"\n{traceback.format_exc()}\"")
                print(message + "\n")
                f.write(message)