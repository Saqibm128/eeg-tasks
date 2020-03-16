from keras import backend as K
import keras.metrics
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def specificity(y_true, y_pred):
    true_negatives = K.shape(y_true)[0] - K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_negatives = K.shape(y_true)[0] - K.sum(K.round(K.clip(y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
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
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    f1_precision = precision(y_true, y_pred)
    f1_recall = recall(y_true, y_pred)
    return 2*((f1_precision*f1_recall)/(f1_precision+f1_recall+K.epsilon()))
