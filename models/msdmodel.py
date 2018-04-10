import sys

from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Add
from layers.biasedadd import BiasedAdd
from layers.inputs import InputChannels
from keras import backend as K

def msdnet(
  num_input_channels,
  width,
  depth,
  num_classes,
  use_dropout=False,
  dropout=0.2,
  conv_block=None,
  merge_op=None,
  filters=1,
  kernel_size=(3,3),
  dilation_rate_fn=None,
  name='msdnet',
  verbose=False
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

  inputs, outputs = __build_model(num_input_channels, filters, kernel_size, merge_op, conv_block, dilation_rate_fn, use_dropout, dropout, width, depth, verbose)
  return Model(inputs=[ inputs ], outputs=[ outputs ])

def __build_input_layer(num_input_channels):
  inputs = Input((None, None, num_input_channels))
  previous_layers = InputChannels(num_input_channels) (inputs)
  return inputs, previous_layers

# TODO: softmax?
def __build_output_layer(previous_layers):
  weighted_feature_maps = []
  for previous_layer in previous_layers:
    weighted_feature_maps.append(Conv2D(1, (1, 1), activation=None, use_bias=False) (previous_layer))
  outputs = BiasedAdd(activation='sigmoid', bias_initializer='random_uniform') (weighted_feature_maps)
  return outputs

def __build_layer(
  filters,
  kernel_size,
  merge_op,
  conv_block,
  dilation_rate_fn,
  use_dropout,
  dropout,
  width, depth,
  i,
  previous_layers
):
  current_layers = []
  for j in range(width):
    current_layers.append(__build_feature_map(filters, kernel_size, conv_block, dilation_rate_fn, use_dropout, dropout, width, depth, i, j, previous_layers))
  previous_layers.extend(current_layers)
  return previous_layers

def __build_feature_map(
  filters,
  kernel_size,
  conv_block,
  dilation_rate_fn,
  use_dropout,
  dropout, 
  width, depth,
  i, j,
  previous_layers
):
  dilated_feature_maps = []
  for previous_layer in previous_layers:
    dilation_rate = dilation_rate_fn(width,i,j)
    dilated_feature_maps.append(Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation=None, use_bias=False) (previous_layer))
    # feature_map = conv_block(filters=filters, kernel_size=kernel_size, width=width, depth=depth, i=i, j=j) (feature_map)
  outputs = BiasedAdd(activation='relu', bias_initializer='random_uniform') (dilated_feature_maps)
  # Dropout if applicable
  if use_dropout:
      outputs = Dropout(dropout) (outputs)
  return outputs

def __build_model(
  num_input_channels,
  filters,
  kernel_size,
  merge_op,
  conv_block,
  dilation_rate_fn,
  use_dropout,
  dropout,
  width, depth,
  verbose
):
  num_total_layers = 2 + depth
  num_current_layers = 0

  def __status_logger(message, num_status_bar):
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    def __log_status(status, num_current_layers, num_total_layers):
      num_bar = int((1.0 * num_current_layers / num_total_layers) * num_status_bar)
      num_spaces = num_status_bar - num_bar

      status_bar = '=' * num_bar
      if num_spaces > 0:
        status_bar = status_bar + '>'
      else:
        status_bar = status_bar + '='

      status_bar = status_bar + ' ' * num_spaces
      status_bar = '[' + status_bar + ']'
      sys.stdout.write(CURSOR_UP_ONE)
      sys.stdout.write(ERASE_LINE)
      sys.stdout.write("\r{}: {}\n{}".format(message, status, status_bar))
      sys.stdout.flush()
    return __log_status
  
  status_logger = __status_logger('Building Model', 40)

  if verbose:
    status_logger('Adding Input Layer', num_current_layers, num_total_layers)
  inputs, previous_layers = __build_input_layer(num_input_channels)
  num_current_layers = num_current_layers + 1

  for i in range(depth):
    num_current_layers = num_current_layers + 1
    if verbose:
      status_logger("Adding Mixed Dense Layer {}/{}".format(i, depth), num_current_layers, num_total_layers)
    previous_layers = __build_layer(filters, kernel_size, merge_op, conv_block, dilation_rate_fn, use_dropout, dropout, width, depth, i, previous_layers)

  num_current_layers = num_current_layers + 1
  if verbose:
    status_logger("Adding Output Layer", num_current_layers, num_total_layers)
  outputs = __build_output_layer(previous_layers)
  if verbose:
    status_logger("Done!", num_total_layers, num_total_layers)
  return inputs, outputs