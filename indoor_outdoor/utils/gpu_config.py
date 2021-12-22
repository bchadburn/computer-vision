import tensorflow as tf
import timeit


def print_system_info():
    def format_header(header, limit=80):
        left_margin = int(round((limit - len(header) - 2) / 2, 0))
        right_margin = int(limit - left_margin - len(header) - 2)
        output = '=' * left_margin + ' ' + header + ' ' + '=' * right_margin
        print(output)

    format_header('Package Versions')
    print('Keras Version:', tf.keras.__version__)
    print('TF Version:', tf.__version__)

    format_header('GPU Details')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        print('GPU Device:', gpu_devices)
        print('GPU Name:', device.name)


def limit_memory(enable=True):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, enable=enable)
    except RuntimeError:
        print('Device can only be configured prior to initialization. Please restart kernel.')


def create_conv_layer():
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return net_cpu


def cpu():
    with tf.device('/cpu:0'):
        net_cpu = create_conv_layer()
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device('/device:GPU:0'):
        net_cpu = create_conv_layer()
        return tf.math.reduce_sum(net_cpu)


def warm_up(devices):
    if not isinstance(devices, list):
        devices = list(devices)
    for device in devices:
        if device == 'GPU':
            gpu()
        elif device == 'CPU':
            cpu()


def get_processor_time(devices=['GPU', 'CPU'], runs=50):
    warm_up(devices)
    print(f'Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
          f'(batch x height x width x channel). Total of {runs} runs.')
    cpu_exp_time = timeit.timeit('cpu()',
                                 number=runs,
                                 setup='from __main__ import cpu')
    print('CPU (s):', round(cpu_exp_time, 2))
    gpu_exp_time = timeit.timeit('gpu()',
                                 number=runs,
                                 setup='from __main__ import gpu')
    print('GPU (s):', round(gpu_exp_time, 2))

    print('GPU speedup over CPU: %.0f%%' % round((cpu_exp_time / gpu_exp_time - 1) * 100, 2))


def check_cpu_gpu():
    print_system_info()
    limit_memory(enable=True)
    get_processor_time()

