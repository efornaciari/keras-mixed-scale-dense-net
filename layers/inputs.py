from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf

class InputChannels(Layer):
  def __init__(
    self,
    num_input_channels,
    **kwargs
  ):

    """
    InputChannels unstacks Input into number of channels specified.
    
    Args:
      num_input_channels : Number of input channels in input tensor.
    """
    super(InputChannels, self).__init__(**kwargs)
    self.num_input_channels = num_input_channels
    self.supports_masking = True

  def build(self, input_shape):
    super(InputChannels, self).build(input_shape)

  def call(self, x):
    """
    Unstacks x into list of tensors.
    
    Args:
      x : Input Tensor to unstack
    
    Returns:
      List of num_input_channels Tensors.
    """
  	return [ K.expand_dims(tensor, axis=-1) for tensor in tf.unstack(x, num=self.num_input_channels, axis=-1) ]

  def compute_output_shape(self, input_shape):
    """
    Computes the output shapes given the input shapes.

    Args:
      input_shape : Shape of input to layer.
    
    Returns:
      Shape of list of num_input_channels Tensors.
    """
    output_shape = []
    single_output_shape = input_shape[:-1] + (1,)
    
    for input_channel in range(self.num_input_channels):
      output_shape.append(single_output_shape)
    return output_shape