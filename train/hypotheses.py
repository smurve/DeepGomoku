import tensorflow as tf
import numpy as np
from tensorflow.feature_column import numeric_column as num



def conv_gomoku(board_size, features, feature_columns, options):

    N = board_size
    
    layout = options['layout']
    
    feature_columns = [num('state', shape=((N+2)*(N+2)*2))]

    input_layer = tf.feature_column.input_layer( 
        features, feature_columns=feature_columns)

    layer = tf.reshape(input_layer, [-1, N+2, N+2, 2], name='reshape_input') 
   
    for filters, kernel in np.reshape(layout, [-1,2]):
        layer = tf.layers.conv2d(inputs=layer, filters=filters, 
                                 kernel_size=[kernel, kernel], strides=[1,1], 
                                 padding='SAME')
        
        beta_l = tf.Variable(-0.5),
        beta_r = tf.Variable(0.5)
        exotic = layer * (layer - beta_l) * (layer - beta_r) * tf.exp(-layer*layer)
        layer = tf.nn.relu(layer)+exotic
        
    layer = tf.layers.conv2d(inputs=layer, filters=1, 
                              kernel_size=[kernel, kernel], strides=[1,1], 
                             padding='SAME')
    
    return layer




def conv_2x1024_5(board_size, features, feature_columns, options):

    N = board_size
    
    feature_columns = [num('state', shape=((N+2)*(N+2)*2))]

    input_layer = tf.feature_column.input_layer( 
        features, feature_columns=feature_columns)

    reshaped = tf.reshape(input_layer, [-1, N+2, N+2, 2], name='reshape_input') 
    
    conva = tf.layers.conv2d(inputs=reshaped, filters=1024, kernel_size=[9,9], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv0 = tf.layers.conv2d(inputs=conva, filters=1024, kernel_size=[9,9], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv1 = tf.layers.conv2d(inputs=conv0, filters=512, kernel_size=[7,7], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[7,7], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[5,5], 
                             strides=[1,1], padding='SAME')

    return conv3


def conv_1024_4(board_size, features, feature_columns, options):

    N = board_size
    
    feature_columns = [num('state', shape=((N+2)*(N+2)*2))]

    input_layer = tf.feature_column.input_layer( 
        features, feature_columns=feature_columns)

    reshaped = tf.reshape(input_layer, [-1, N+2, N+2, 2], name='reshape_input') 
    
    conv0 = tf.layers.conv2d(inputs=reshaped, filters=1024, kernel_size=[9,9], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv1 = tf.layers.conv2d(inputs=conv0, filters=512, kernel_size=[9,9], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[7,7], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[5,5], 
                             strides=[1,1], padding='SAME')

    return conv3


def conv_512_3(board_size, features, feature_columns, options):

    N = board_size
    
    feature_columns = [num('state', shape=((N+2)*(N+2)*2))]

    input_layer = tf.feature_column.input_layer( 
        features, feature_columns=feature_columns)

    reshaped = tf.reshape(input_layer, [-1, N+2, N+2, 2], name='reshape_input') 
    
    conv1 = tf.layers.conv2d(inputs=reshaped, filters=512, kernel_size=[9,9], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[7,7], 
                             strides=[1,1], padding='SAME', activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[5,5], 
                             strides=[1,1], padding='SAME')

    return conv3


