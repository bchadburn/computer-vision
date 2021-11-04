import tensorflow as tf
import timeit
import functools
import time
import os
import argparse


def create_directory(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def dir_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"image_name:{path} is not a valid path")


def verify_create_paths(path):
    if type(path) is list:
        for i in path:
            if not os.path.exists(i):
                os.makedirs(i)
    elif type(path) is str:
        if not os.path.exists(path):
            os.makedirs(path)


def time_it(func):
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


def print_system_info():
    def format_header(header, limit=79):
        Left_margin = int(round((limit - len(header) - 2) / 2, 0))
        Right_margin = int(limit - Left_margin - len(header) - 2)
        output = "=" * (Left_margin) + " " + header + " " + "=" * (Right_margin)
        print(output)

    format_header("Package Versions")
    print("Keras Version:", tf.keras.__version__)
    print("TF Version:", tf.__version__)
    print()

    format_header("GPU Details")
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_devices:
        print("GPU Device:", gpu_devices)
        print("GPU Name:", gpu.name)
    print()


def limit_memory(enable=True):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, enable=enable)
    except (RuntimeError):
        print("Device can only be configured prior to initialization. Please restart kernal.")


def cpu():
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device('/device:GPU:0'):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


def warm_up(devices):
    if type(devices) != list:
        devices = list(devices)
    for device in devices:
        if device == "GPU":
            gpu()
        elif device == "CPU":
            cpu()


def get_processor_time(devices=["GPU", "CPU"], runs=10):
    warm_up(devices)
    print(f'Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
          '(batch x height x width x channel). Sum of {runs} runs.')
    cpu_exp_time = timeit.timeit('cpu()',
                                 number=runs,
                                 setup="from __main__ import cpu")
    print('CPU (s):', round(cpu_exp_time, 2))
    gpu_exp_time = timeit.timeit('gpu()',
                                 number=runs,
                                 setup="from __main__ import gpu")
    print('GPU (s):', round(gpu_exp_time, 2))

    print("GPU speedup over CPU: %.0f%%" % round((cpu_exp_time / gpu_exp_time - 1) * 100, 2))


def check_cpu_gpu():
    print_system_info()
    limit_memory(enable=True)
    get_processor_time()
