import pickle
import tensorflow as tf

test_filenames=["/n/scratch2/ms994/medium_size/test_1.tfrecords","/n/scratch2/ms994/medium_size/test_2.tfrecords","/n/scratch2/ms994/medium_size/test_3.tfrecords","/n/scratch2/ms994/medium_size/test_0.tfrecords"]
def read_tfrecord(example):
    features = {'original_index': tf.io.FixedLenFeature([1], tf.int64, ),\
               'data':  tf.FixedLenFeature([9*21*1000], tf.float32,),\
               'label':  tf.FixedLenFeature([10], tf.int64, [0 for i in range(10)]),\
               'subtypeLabel':  tf.FixedLenFeature([10], tf.int64, [0 for i in range(10)]),\
               'patient':  tf.FixedLenFeature([1], tf.int64,), \
               'session':  tf.FixedLenFeature([1], tf.int64,),
                       }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
#     return example

    data = tf.reshape(example['data'], [9,21,1000,1])
    # data = (example['data'])

    class_label = tf.cast(example['label'], tf.int32)


    return data, tf.one_hot(class_label[:9], 2)
def get_batched_dataset(filenames=test_filenames, batch_size=4, use_fft=False, max_queue_size=40, max_std=100, n_process=4, is_train=True):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=n_process)
    if is_train and max_std != None:
        dataset = dataset.filter(lambda x, y: tf.reduce_all(tf.math.reduce_std(x, axis=0) < max_std))
    if use_fft:
        dataset = dataset.map(lambda x, y: ( \
                                            tf.cast(tf.signal.fft( tf.cast(x, tf.complex64)), tf.float64), \
                                            y), num_parallel_calls=n_process)
    dataset = dataset.repeat()
    if is_train:
        dataset = dataset.shuffle(128)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(max_queue_size)
    return dataset
train_filenames=["/n/scratch2/ms994/medium_size/train_1.tfrecords","/n/scratch2/ms994/medium_size/train_2.tfrecords","/n/scratch2/ms994/medium_size/train_3.tfrecords","/n/scratch2/ms994/medium_size/train_0.tfrecords"]

def get_test_dataset(test_filenames=test_filenames):
    return get_batched_dataset(test_filenames, is_train=False)
def get_train_dataset(test_filenames=train_filenames):
    return get_batched_dataset(test_filenames, is_train=False)

test = get_test_dataset()
train = get_train_dataset()
import sys, os
sys.path.append(os.path.realpath(".."))
from keras_models.metrics import f1, sensitivity, specificity
model = tf.keras.models.load_model("/n/scratch2/ms994/out/WGBYHTDBVOQAXBVX.h5", custom_objects={"f1":f1,"sensitivity":sensitivity,"specificity":specificity})

testPred = model.predict(test, steps=700)
test = get_test_dataset()
print(model.evaluate(test, steps=700))

trainPred = model.predict(train, steps=700)
train = get_train_dataset()
print(model.evaluate(train, steps=700))
import pickle
pickle.dump((testPred, trainPred), open("validation.pkl", "rb"))
