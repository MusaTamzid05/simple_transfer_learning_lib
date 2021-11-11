import tensorflow as tf

def limit_gpu(memory = 2048):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit = memory)])
      except RuntimeError as e:
        print(e)
