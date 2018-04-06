from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Add
from layers.biasedadd import BiasedAdd

def msdnet(
  width,
  depth,
  num_classes,
  conv_block=None,
  merge_op=None,
  filters=1,
  kernel_size=(3,3),
  dilation_rate_fn=None,
  name = 'msdnet',
):
  """ Construct a Mixed-Scale Dense Net

  Args
    inputs            : 
    num_classes       : 
    conv_block        : 
    dilation_rate_fn  : 
    name              : 
  """
  if conv_block is None:
    def __conv_block(**kwargs):
      return Conv2D(filters, kernel_size, padding='same', activation='elu')
    conv_block = __conv_block

  if dilation_rate_fn is None:
    def __dilation_rate_fn(width, i, j):
      return (i * width + j) % 10 + 1
    dilation_rate_fn = __dilation_rate_fn

  if merge_op is None:
    def __merge_op():
      return Add()
    merge_op = __merge_op

  inputs, outputs = __build_model(filters, kernel_size, merge_op, conv_block, dilation_rate_fn, width, depth)
  return Model(inputs=[ inputs ], outputs=[ outputs ])

def __build_input_layer():
  inputs = Input((None, None, 3))
  previous_layer = Lambda(lambda x: x / 255) (inputs)
  return inputs, [ previous_layer ]

# TODO: softmax?
def __build_output_layer(previous_layers):
  weighted_feature_maps = []
  for previous_layer in previous_layers:
    weighted_feature_maps.append(Conv2D(1, (1, 1), activation=None, use_bias=False) (previous_layer))
  outputs = BiasedAdd(activation='sigmoid', bias_initializer='random_uniform') (weighted_feature_maps)
  return outputs

def __build_layer(filters, kernel_size, merge_op, conv_block, dilation_rate_fn, width, depth, i, previous_layers):
  current_layers = []
  for j in range(width):
    current_layers.append(__build_feature_map(filters, kernel_size, conv_block, dilation_rate_fn, width, depth, i, j, previous_layers))
  previous_layers.extend(current_layers)
  return previous_layers

def __build_feature_map(filters, kernel_size, conv_block, dilation_rate_fn, width, depth, i, j, previous_layers):
  dilated_feature_maps = []
  for previous_layer in previous_layers:
    dilated_feature_maps.append(Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate_fn(width,i,j), activation=None, use_bias=False) (previous_layer))
    # feature_map = conv_block(filters=filters, kernel_size=kernel_size, width=width, depth=depth, i=i, j=j) (feature_map)
  outputs = BiasedAdd(activation='relu', bias_initializer='random_uniform') (dilated_feature_maps)
  return outputs

def __build_model(filters, kernel_size, merge_op, conv_block, dilation_rate_fn, width, depth):
  inputs, previous_layers = __build_input_layer()
  
  for i in range(depth):
    previous_layers = __build_layer(filters, kernel_size, merge_op, conv_block, dilation_rate_fn, width, depth, i, previous_layers)

  outputs = __build_output_layer(previous_layers)
  return inputs, outputs