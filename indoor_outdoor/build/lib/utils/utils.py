import functools
import time
import os
import argparse


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def dir_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file path")


def verify_create_paths(*paths):
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)


def time_run(func):
    """Decorator for printing function runtime"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Completed {func.__name__!r} in {run_time:.2f} secs")
        return result

    return wrapper_timer
