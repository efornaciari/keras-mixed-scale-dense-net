from keras import backend as K
from keras.layers.merge import Add, Concatenate

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils

class BiasedAdd(Add):

  def __init__(
    self,
    data_format=None,
    activation=None,
    use_bias=True,
    bias_initializer='zeros',
    bias_regularizer=None,
    bias_constraint=None,
    **kwargs
  ):

    self.data_format = conv_utils.normalize_data_format(data_format)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)

    super(BiasedAdd, self).__init__(**kwargs)
    self.supports_masking = True

  def build(self, input_shape):
    # Used purely for shape validation.
    if not isinstance(input_shape, list):
        input_shape = [ input_shape ]
        # raise ValueError('A merge layer should be called on a list of inputs.')

    batch_sizes = [s[0] for s in input_shape if s is not None]
    batch_sizes = set(batch_sizes)
    batch_sizes -= set([None])
    if len(batch_sizes) > 1:
        raise ValueError('Can not merge tensors with different '
                         'batch sizes. Got tensors with shapes : ' +
                         str(input_shape))
    if input_shape[0] is None:
        output_shape = None
    else:
        output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
        if input_shape[i] is None:
            shape = None
        else:
            shape = input_shape[i][1:]
        output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    # If the inputs have different ranks, we have to reshape them
    # to make them broadcastable.
    if None not in input_shape and len(set(map(len, input_shape))) == 1:
        self._reshape_required = False
    else:
        self._reshape_required = True


    if self.use_bias:
        self.bias = self.add_weight(shape=(input_shape[0][-1],),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True)
    else:
        self.bias = None

  def _merge_function(self, inputs):
    outputs = super(BiasedAdd, self)._merge_function(inputs)

    # Apply Bias
    if self.use_bias:
        outputs = K.bias_add(
            outputs,
            self.bias,
            data_format=self.data_format)

    # Apply Activation
    if self.activation is not None:
        outputs = self.activation(outputs)

    return outputs

  def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, list):
        input_shape = [ input_shape ]
    if input_shape[0] is None:
        output_shape = None
    else:
        output_shape = input_shape[0][1:]
    for i in range(1, len(input_shape)):
        if input_shape[i] is None:
            shape = None
        else:
            shape = input_shape[i][1:]
        output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
    batch_sizes = [s[0] for s in input_shape if s is not None]
    batch_sizes = set(batch_sizes)
    batch_sizes -= set([None])
    if len(batch_sizes) == 1:
        output_shape = (list(batch_sizes)[0],) + output_shape
    else:
        output_shape = (None,) + output_shape
    return output_shape
