import time


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')