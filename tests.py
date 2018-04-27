from main import main, POSSIBLE_MODELS, POSSIBLE_TASKS
from pathlib import Path
import datetime

folder = Path("test_logs")
folder.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

with open(folder/f"{timestamp}.log", "w") as f:
    for task in POSSIBLE_TASKS:
        for model in POSSIBLE_MODELS:
            try:
                main(["--task", task, "-pmodel_type=" + model,
                  "-pnum_batches=100", "-pbatch_size=2",
                  "--checkpoint_interval", "50", "--report_interval", "50"])
                message = f"SUCCESS - Task: {task}, Model: {model}"
                f.write(message)
            except Exception as err:
                message = f"FAILED - Task: {task}, Model: {model}, Error: '{vars(err)}'"
                print(message)
                f.write(f"FAILED - Task: {task}, Model: {model}, Error: '{vars(err)}'")