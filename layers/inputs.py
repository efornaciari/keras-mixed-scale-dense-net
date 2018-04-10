from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf

class InputChannels(Layer):

  def __init__(
    self,
    input_channels,
    **kwargs
  ):
    super(InputChannels, self).__init__(**kwargs)
    self.input_channels = input_channels
    self.supports_masking = True

  def build(self, input_shape):
    super(InputChannels, self).build(input_shape)

  def call(self, x):
  	return [ K.expand_dims(tensor, axis=-1) for tensor in tf.unstack(x, num=self.input_channels, axis=-1) ]

  def compute_output_shape(self, input_shape):
    output_shape = []
    single_output_shape = input_shape[:-2] + (1,)
    
    for input_channel in range(self.input_channels):
      output_shape.append(single_output_shape)
    return output_shape