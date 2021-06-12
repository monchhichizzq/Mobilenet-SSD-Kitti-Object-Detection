import tensorflow as tf
import numpy as np
from prettytable import PrettyTable


def measure_weights_sparsity(model):
    save_file = open('weight_sparsity.txt', 'w+')
    w_spar_table = PrettyTable()
    w_spar_table.field_names = ['layer name', 'weights(k)', 'zero_weights(k)', 'sparsity(%)']
    target_layers = ['Conv2D', 'DepthwiseConv2D', 'PruneLowMagnitude', 'GConv2D', 'GDepthwiseConv2D']
    # target_layers = ['PruneLowMagnitude']
    total_weights = tf.constant(0, dtype=tf.int64)
    non_zero_weights = tf.constant(0, dtype=tf.int64)
    for i, layer in enumerate(model.layers):

        if layer.__class__.__name__ in target_layers:
            layer_weights = layer.get_weights()
            # print(np.shape(layer_weights))
            # for i, w in enumerate(layer_weights):
            #     print(i, np.shape(w))
            # len(np.shape(layer_weights)) == 2 means there is a bias
            layer_weights = layer_weights[1]
            non_zero_neurons = tf.math.count_nonzero(layer_weights)
            all_neurons = tf.reduce_prod(tf.shape(layer_weights))
            all_neurons = tf.cast(all_neurons, dtype=tf.int64)
            w_sparsity = tf.round(tf.divide((all_neurons - non_zero_neurons), all_neurons) * 100)
            #print(all_neurons, non_zero_weights)
            #print('{0} weight sparsity: {1}%, all weights {2} K'.format(layer.name, w_sparsity, all_neurons/1000))
            w_spar_table.add_row(
                [layer.name, all_neurons.numpy() / 1000, (all_neurons.numpy() - non_zero_neurons.numpy()) / 1000,
                 w_sparsity.numpy()])

            non_zero_weights = tf.math.add(non_zero_neurons, non_zero_weights)
            total_weights = tf.math.add(all_neurons, total_weights)

    # structure pruning
    total_weights_M = tf.round(tf.cast(7.23, dtype=tf.float64))
    zero_weights_M = tf.round(tf.divide((- non_zero_weights + tf.cast(7.23 * 10 ** 6, dtype=tf.int64)), 10 ** 6))
    net_sparsity = tf.round(tf.divide(zero_weights_M, total_weights_M) * 100)
    # net_sparsity = tf.round(tf.divide((total_weights - non_zero_weights), total_weights) * 100)
    # zero_weights = tf.divide((total_weights - non_zero_weights), 10**6)
    # total_weights = tf.divide(total_weights, 10**6)
    # print('\nzero weights: {0} M, total weights: {1} M'.format(zero_weights_M, total_weights_M))
    # print('Network weight sparsity: {0}%\n'.format(net_sparsity))
    w_spar_table.add_row(
        ['Total', str(total_weights_M.numpy()) + ' M', str(zero_weights_M.numpy()) + ' M', net_sparsity.numpy()])
    print(w_spar_table)
    print(w_spar_table, file=save_file)
    save_file.close()
    return net_sparsity.numpy()


# Add Weight_Quantization_Callback for training
class Weight_Sparsity_Callback(tf.keras.callbacks.Callback):

    def __init__(self):
        super(Weight_Sparsity_Callback, self).__init__()
        pass

    def on_train_begin(self, logs=None):
        print('Train begin, show layer sparsity!')
        measure_weights_sparsity(self.model)

    def on_epoch_end(self, batch, logs=None):
        print('Test begin, show layer sparsity!')
        measure_weights_sparsity(self.model)

    def on_predict_begin(self, logs=None):
        print('Prediction begin, show layer sparsity!')
        measure_weights_sparsity(self.model)
