from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K

#from https://github.com/yaringal/multi-task-learning-example

# Custom loss layer
class HomeoschedasticMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, loss_funcs=[], multiplier=None, **kwargs):
        #multiplier is used to change signs
        self.nb_outputs = nb_outputs
        self.output_dim = nb_outputs
        self.is_placeholder = True
        if multiplier is None:
          self.multiplier = [1 for i in range(nb_outputs)]
        else:
          self.multiplier = multiplier
        self.loss_funcs = loss_funcs
        super(HomeoschedasticMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(HomeoschedasticMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for i, zipped_args in enumerate(zip(ys_true, ys_pred, self.log_vars)):
            y_true, y_pred, log_var = zipped_args
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * self.multiplier[i] * self.loss_funcs[i](y_true, y_pred)**2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # pass thru the predictions
        return K.concatenate(ys_pred)

    # def compute_output_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     outputs = input_shape[self.nb_outputs:]
    #
    #     return outputs
