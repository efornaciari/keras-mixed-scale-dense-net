import sys

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D

from layers.biasedadd import BiasedAdd
from layers.inputs import InputChannels

from keras import backend as K

def msdnet(
  num_input_channels,
  network_shape,
  num_output_classes,
  use_dropout=False,
  intermediate_activation="relu",
  output_activation="softmax",
  dropout=0.2,
  filters=1,
  kernel_size=(3,3),
  dilation_rate_fn=None,
  verbose=False,
  **kwargs
):
  """ Construct a Mixed-Scale Dense Net

  Args
    inputs            : 
    num_classes       : 
    conv_block        : 
    dilation_rate_fn  : 
  """

  width, depth = network_shape

  if dilation_rate_fn is None:
    def __dilation_rate_fn(width, i, j):
      return (i * width + j) % 10 + 1
    dilation_rate_fn = __dilation_rate_fn

  inputs, outputs = __build_model(
    num_input_channels,
    num_output_classes,
    filters,
    kernel_size,
    dilation_rate_fn,
    use_dropout,
    dropout,
    width, depth,
    verbose)
  return Model(inputs=[ inputs ], outputs=[ outputs ])

def __build_input_layer(num_input_channels):
  inputs = Input((None, None, num_input_channels))
  previous_layers = InputChannels(num_input_channels) (inputs)
  return inputs, previous_layers

# TODO: softmax?
def __build_output_layer(num_classes, previous_layers):
  weighted_feature_maps = []
  for previous_layer in previous_layers:
    weighted_feature_maps.append(Conv2D(1, (1, 1), activation=None, use_bias=False) (previous_layer))
  outputs = BiasedAdd(bias_initializer='random_uniform', activation='relu') (weighted_feature_maps)
  outputs = Conv2D(num_classes, (1, 1), activation='softmax', use_bias=False) (outputs)
  return outputs

def __build_layer(
  filters,
  kernel_size,
  dilation_rate_fn,
  use_dropout,
  dropout,
  width, depth,
  i,
  previous_layers
):
  current_layers = []
  for j in range(width):
    current_layers.append(
      __build_feature_map(
        filters,
        kernel_size,
        dilation_rate_fn,
        use_dropout, dropout,
        width, depth,
        i, j,
        previous_layers))
  previous_layers.extend(current_layers)
  return previous_layers

def __build_feature_map(
  filters,
  kernel_size,
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
  outputs = BiasedAdd(bias_initializer='random_uniform', activation='relu') (dilated_feature_maps)
  # Dropout if applicable
  if use_dropout:
      outputs = Dropout(dropout) (outputs)
  return outputs

def __build_model(
  num_input_channels,
  num_classes,
  filters,
  kernel_size,
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
    previous_layers = __build_layer(filters, kernel_size, dilation_rate_fn, use_dropout, dropout, width, depth, i, previous_layers)

  num_current_layers = num_current_layers + 1
  if verbose:
    status_logger("Adding Output Layer", num_current_layers, num_total_layers)
  outputs = __build_output_layer(num_classes, previous_layers)
  if verbose:
    status_logger("Done!", num_total_layers, num_total_layers)
  return inputs, outputs