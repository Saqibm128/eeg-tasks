import tensorflow as tf
def read_tfrecord(example):
    features = { \
               'data':  tf.io.FixedLenFeature([21*1000], tf.float32,),\
               'label':  tf.io.FixedLenFeature([1], tf.int64,),\
               'subtypeLabel':  tf.io.FixedLenFeature([1], tf.int64,),\
               'session':  tf.io.FixedLenFeature([1], tf.int64,), \
               'montage':  tf.io.FixedLenFeature([22], tf.int64,)}

    # decode the TFRecord
    return tf.io.parse_single_example(example, features)

def read_tfrecord_autoencoder(example):
    example = read_tfrecord(example)
    return tf.reshape(example["data"], (1000,21,1)), tf.reshape(example["data"], (1000,21,1))

def get_batched_dataset(filenames, map_function=None, batch_size=64, max_queue_size=10,  n_process=4, is_train=False):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
    dataset = dataset.map(map_function, num_parallel_calls=n_process)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(256)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if is_train:
        dataset = dataset.prefetch(max_queue_size)
    else:
        dataset = dataset.prefetch(int(max_queue_size/4))
    return dataset
