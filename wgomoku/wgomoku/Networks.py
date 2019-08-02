import tensorflow as tf


class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, n_filters, activation):
        super(InceptionLayer, self).__init__()
        
        self.l3 = tf.keras.layers.Conv2D(kernel_size=3, filters=n_filters, 
                                  padding='same', activation=activation)
        
        self.l5 = tf.keras.layers.Conv2D(kernel_size=5, filters=n_filters,
                                  padding='same', activation=activation)
        
    def call(self, inputs):        
        inputs = tf.cast(inputs, dtype=tf.float32)
        return tf.concat([self.l3(inputs), self.l5(inputs)], axis=3)

    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_filters, activation):
        super(ResidualBlock, self).__init__()
        
        self.skip = tf.keras.layers.Conv2D(kernel_size=1, filters=2,
                                padding='same', activation=None)

        self.layers = [InceptionLayer(n_filters, activation=activation)
                      for i in range(n_layers)]
        
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32)
        for layer in self.layers:
            x = layer(x)
        return self.bn(tf.concat([self.skip(x), x], axis=3))

    
class PolicyModel(tf.keras.Model):
    def __init__(self, board_size, n_blocks, n_layers, n_filters, activation):
        super(PolicyModel, self).__init__()
        self.board_size=board_size
        
        self.blocks = [ResidualBlock(n_layers, n_filters, activation)
                       for _ in range(n_blocks)]
        
        self.logits = tf.keras.layers.Conv2D(kernel_size=3, filters=1, strides=1,
               padding='valid', activation='tanh')
    
    
    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32)
        for block in self.blocks:
            x = block(x)
        N = self.board_size
        return tf.reshape(self.logits(x), [-1, N*N])

    
    def dist(self, inputs):
        N = self.board_size
        dist = tf.keras.activations.softmax(self.call(inputs), axis=[1,2])
        return tf.reshape(dist, [N, N])

 