from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization, Activation, Add, UpSampling2D

def msdnet(
  width,
  depth,
  num_classes,
  conv_block,
  filters=3,
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

  inputs, outputs = __build_model(filters, kernel_size, width, depth)
  return Model(inputs=[ inputs ], outputs=[ outputs ])

def __build_input_layer():
  inputs = Input((None, None, 3))
  previous_layer = Lambda(lambda x: x / 255) (inputs)
  return inputs, previous_layer

# TODO: softmax?
def __build_output_layer(previous_layer):
  outputs = Conv2D(1, (1, 1), activation='sigmoid') (previous_layer)
  return outputs

def __build_layer(filters, kernel_size, width, depth, i, previous_layer):
  #print "Building Layer {} of {}...".format(i, depth)
  current_layers = [ previous_layer ]
  for j in range(width):
    current_layers.append(__build_node(filters, kernel_size, width, depth, i, j, previous_layer))
  return Concatenate(axis=-1) (current_layers)

def __build_node(filters, kernel_size, width, depth, i, j, previous_layer):
  #print "\tBuilding Node {} of {}...".format(j, width)
  node = Conv2D(filters, kernel_size, padding='same', dilation_rate=2 ** j, activation='elu', use_bias=True) (previous_layer)
  node = Conv2D(filters, kernel_size, padding='same', activation='elu') (node)
  return node

def __build_model(filters, kernel_size, width, depth):
  inputs, previous_layer = __build_input_layer()
  
  for i in range(depth):
    previous_layer = __build_layer(filters, kernel_size, width, depth, i, previous_layer)
    
  outputs = __build_output_layer(previous_layer)
  return inputs, outputs