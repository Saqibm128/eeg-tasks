from keras import backend as K
import keras.metrics
import tensorflow as tf

from sklearn.metrics import roc_auc_score

def non_error_roc_auc_score(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.0

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

def auroc(y_true, y_pred):
    return tf.py_func(non_error_roc_auc_score, (y_true, y_pred), tf.double)

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(K.argmax(y_true) * K.argmax(y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(K.argmax(y_true), 0, 1)))
    recall = tf.cast(true_positives, tf.float64) / (tf.cast(possible_positives, tf.float64) + K.epsilon())
    return recall

def specificity(y_true, y_pred):
    true_negatives =  K.sum(K.round(K.clip((1-K.argmax(y_true)) * (1-K.argmax(y_pred)), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip((1-K.argmax(y_pred)), 0, 1)))
    specificity = tf.cast(true_negatives, tf.float64) / (tf.cast(possible_negatives, tf.float64) + K.epsilon())
    return specificity
    # return keras.metrics.SpecificityAtSensitivity(sensitivity(y_true, y_pred))

def sensitivity(y_true, y_pred):
    return recall(y_true, y_pred)

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(K.argmax(y_true) * K.argmax(y_pred), 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(K.argmax(y_pred), 0, 1)))
    precision = tf.cast(true_positives, tf.float64) / (tf.cast(predicted_positives, tf.float64) +K.epsilon())
    return precision

def f1(y_true, y_pred):
    f1_precision = precision(y_true, y_pred)
    f1_recall = recall(y_true, y_pred)
    return 2*((f1_precision*f1_recall)/(tf.cast(f1_precision, tf.float64) + tf.cast(f1_recall, tf.float64) +K.epsilon()))
