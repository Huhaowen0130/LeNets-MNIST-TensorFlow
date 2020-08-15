# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

# LeNet-1
def LeNet1(x):
    # 5x5x4 Convolution with ReLU
    x1 = layers.conv2d(x, 4, [5, 5], padding='VALID')
    
    # 2x2 Average Pooling
    x2 = tf.nn.avg_pool(x1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    # 5x5x12 Convolution with ReLU
    x3 = layers.conv2d(x2, 12, [5, 5], padding='VALID')
    
    # 2x2 Average Pooling
    x4 = tf.nn.avg_pool(x3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    # Flatten
    x4 = layers.flatten(x4)
    
    # 192x10 Fully Connection with Softmax
    # for softmax_cross_entropy_with_logits used as loss, activation_fn should be None here
    out = layers.fully_connected(x4, 10, activation_fn=None)
    
    return out

# LeNet-4
def LeNet4(x):
    # 5x5x4 Convolution with ReLU
    x1 = layers.conv2d(x, 4, [5, 5], padding='VALID')
    
    # 2x2 Average Pooling
    x2 = tf.nn.avg_pool(x1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    # 5x5x16 Convolution with ReLU
    x3 = layers.conv2d(x2, 16, [5, 5], padding='VALID')
    
    # 2x2 Average Pooling
    x4 = tf.nn.avg_pool(x3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    
    # Flatten
    x4 = layers.flatten(x4)
    
    # 256x120 Fully Connection with ReLU
    x5 = layers.fully_connected(x4, 120)
    
    # 120x10 Fully Connection with Softmax
    # for softmax_cross_entropy_with_logits used as loss, activation_fn should be None here
    out = layers.fully_connected(x5, 10, activation_fn=None)
    
    return out

# LeNet-5
def LeNet5(x):
    # C1 Layer: 5x5x6 Convolution with ReLU
    c1 = layers.conv2d(x, 6, [5, 5], padding='VALID')
    
    # S2 Layer: 2x2 Average Pooling, 6 weights and 6 biases(Batch Normalization with Sigmoid)
    s2 = tf.nn.avg_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    s2 = layers.batch_norm(s2, activation_fn=tf.nn.sigmoid)
    
    # C3 Layer: implemented with 5x5x16 Convolution with ReLU
    c3 = layers.conv2d(s2, 16, [5, 5], padding='VALID')
    
    # S4 Layer: 2x2 Average Pooling, 6 weights and 6 biases(Batch Normalization with Sigmoid)
    s4 = tf.nn.avg_pool(c3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    s4 = layers.batch_norm(s4, activation_fn=tf.nn.sigmoid)
    
    # C5 Layer: 5x5x120 Convolution with ReLU
    c5 = layers.conv2d(s4, 120, [5, 5], padding='VALID')
    
    # Flatten
    c5 = layers.flatten(c5)
    
    # F6 Layer: 120x84 Fully Connection with ReLU
    f6 = layers.fully_connected(c5, 84)
    
    # OUTPUT Layer: implemented with 84x10 Fully Connection with Softmax
    # for softmax_cross_entropy_with_logits used as loss, activation_fn should be None here
    out = layers.fully_connected(f6, 10, activation_fn=None)
    
    return out