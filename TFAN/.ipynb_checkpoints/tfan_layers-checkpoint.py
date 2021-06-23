# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 14:39:51 2021

@author: COMPREDICT
"""

# +
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dropout, Layer
from tensorflow.keras.layers import Conv1D, TimeDistributed, Dense
import tensorflow as tf

from tensorflow import nn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec


from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export


class residual_block(Layer):
    """ Residual Block consisting of:
    DepthwiseConv1D->Dropout->DepthwiseConv1D->Dropout->Residual Connection
    Dilation is constant within residual block.

        Input shape:
            3D Tensor with shape:
             [batch, Tx, n features]
        Output shape:
            3D Tensor with shape:
            [batch, Tx, n features*depth_multiplier]
    """
    def __init__(self, depthwise_kernel_size, dilation_rate=1,
                 padding="causal", activation=tf.keras.activations.swish,
                 dropout=0.2, depth_multiplier=1,
                 depthwise_initializer=tf.keras.initializers.HeUniform(),
                 **kwargs):
        super(residual_block, self).__init__()
        self.dropout = Dropout(dropout)
        self.conv1 = DepthwiseConv1D(
                depthwise_kernel_size,
                dilation_rate=dilation_rate,
                name='depthwiseConv1D_dilation_{}_0'.format(dilation_rate),
                padding=padding, activation=activation,
                depth_multiplier=depth_multiplier,
                depthwise_initializer=depthwise_initializer)
        self.conv2 = DepthwiseConv1D(
                depthwise_kernel_size,
                dilation_rate=dilation_rate,
                name='depthwiseConv1D_dilation_{}_1'.format(dilation_rate),
                padding=padding, activation=activation,
                depth_multiplier=depth_multiplier,
                depthwise_initializer=depthwise_initializer)
        self.conv1x1 = DepthwiseConv1D(
                1, padding=padding,
                activation=tf.keras.activations.linear,
                depth_multiplier=depth_multiplier,
                kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.depth_multiplier = depth_multiplier
        self.dilation_rate = dilation_rate

    def call(self, x):

        # get dimensions

        batch_size = tf.cast(tf.shape(x)[0], dtype=tf.int32)
        Tx = tf.cast(tf.shape(x)[1], dtype=tf.int32)

        # get residual

        """
        For the case of depth_multiplier != 1:
        x before and after the convolution will have a different amount of
        channels. Thus, a 1x1 convolution is applied. This is relevant only
        for the first residual block with dilation_rate = 1.
        """
        if self.dilation_rate == 1 and self.depth_multiplier > 1:
            n_features = tf.cast(tf.shape(x)[2], dtype=tf.int32)
            x_res = self.conv1x1(x)
            # shape == (batch, Tx, n_features*depth_multiplier)
        else:
            x_res = x  # 1x1 Convolution not needed
            n_features = tf.cast(tf.shape(x)[2]/self.depth_multiplier,
                                 dtype=tf.int32)

        # apply first conv

        x = self.conv1(x)  # shape == (batch, Tx, n_features*depth_multiplier)
        """
        DepthwiseConv1D does not support having multiple channels per feature
        as input. Thus, the convolution is applied to x
        [batch, Tx, n features*channels] rather than
        [batch, Tx, n features, channels]. To compensate, the result of this
        convolution has to be reshaped after the convolution. This is not the
        case for the very first convolution, as the input is x
        [batch, Tx, n features].
        """
        multi_channel = False
        # if not the first residual block, apply multi-channel computation in
        # first conv
        if self.dilation_rate > 1:
            multi_channel = True
        if multi_channel:
            x = tf.reshape(x, (batch_size, Tx, n_features,
                               self.depth_multiplier, self.depth_multiplier))
            x = tf.reduce_sum(x, axis=-2)
            x = tf.reshape(x, (batch_size, Tx,
                               n_features*self.depth_multiplier))
        multi_channel = True
        # apply activation
        x = tf.keras.activations.swish(x)
        x = self.dropout(x)

        # second convolution

        x = self.conv2(x)

        if multi_channel:
            x = tf.reshape(x, (batch_size, Tx, n_features,
                               self.depth_multiplier, self.depth_multiplier))
            x = tf.reduce_sum(x, axis=-2)
            x = tf.reshape(x, (batch_size, Tx,
                               n_features*self.depth_multiplier))
        # apply activation
        x = tf.keras.activations.swish(x)
        x = self.dropout(x)

        # residual connection

        out = x + x_res

        return out


class deal_with_tf_bug(Layer):
    """ If this layer is not used, tf will complain about the shapes not being known in the next layer and the model can
    only be fit if run_eagerly=True. Unfortunately, no other fix could be found. This layer applies some dummy
    computations that will not affect the output of the residual blocks.
    """

    def __init__(self, Tx):
        super(deal_with_tf_bug, self).__init__()
        self.wa = Dense(Tx, trainable=False)
        self.wb = Dense(Tx, trainable=False)
        self.Tx = Tx

    def call(self, x):
        keys = self.wa(x)
        queries = self.wb(x)
        scores = keys * queries
        dk = tf.math.sqrt(tf.cast(self.Tx, dtype=tf.float32))
        scores = scores / dk
        att = tf.math.divide_no_nan(scores + 0.0000001, scores + 0.0000001)
        out = att * x

        return out


class MultiHeadAttention(Layer):
    """
    Multi-Head-Attention as described in Attention is all you need (2017).

        Input shape:
            3D Tensor with shape:
             [batch, n features, Tx]
        Output shape:
             out: 3D tensor of shape == (batch, n features, d_model)
             att_weights: attention weights of shape ==
             (batch, num_heads, n features, n features)
             values: values of shape == (batch, num_heads, n features,
                                         d_model, depth)
    """

    def __init__(self, d_model, num_heads, regularization, p):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.regularization = regularization
        self.p = p

        # assert that d_model can be split into equally sized heads
        assert self.d_model % self.num_heads == 0

        # depth of each head is d_model/num_heads
        # -> overall computational cost is the same
        self.depth = self.d_model // self.num_heads

        self.wq = TimeDistributed(Dense(
                d_model,
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
        self.wk = TimeDistributed(Dense(
                d_model,
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
        self.wv = TimeDistributed(Dense(
                d_model,
                kernel_initializer=tf.keras.initializers.GlorotUniform()))

        self.dense = TimeDistributed(Dense(
                d_model,
                kernel_initializer=tf.keras.initializers.GlorotUniform()))

    def split_heads(self, x, batch_size, n_features):
        """ Split the last dimension (d_model) into (num_heads, depth).
            Transpose to (batch, num_heads, features, depth).
        """
        x = tf.reshape(x, (batch_size, n_features, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def dropAttention(self, attention_weights, p):
        """
        Regularization method for attention following the paper:
        https://arxiv.org/pdf/1907.11065.pdf
        With a probability of p a column of the attention weights is set to 0.
        This leads to a feature in the values v being ignored with a
        probability of p during training. An individual mask is generated for
        each sample in a batch, whereas the same mask is used across different
        heads.

        Args:
            attention weights: shape == (batch_size, num_heads, features,
                                         features)
            p: probability to set a column to 0

        Returns:
            attention_weights
        """

        batch_size = tf.shape(attention_weights)[0]
        n_features = tf.shape(attention_weights)[-1]
        # create mask that sets columns to 0 with a certain probability
        # different mask for every sample in batch
        mask = tf.random.uniform(shape=(batch_size, n_features)) > p
        mask = tf.cast(mask, dtype='float32')
        mask = tf.reshape(mask, [batch_size, 1, 1, -1])
        # mask attention weights
        attention_weights = attention_weights*mask
        # normalized rescaling of attention_weights to ensure that attention
        # weights in every row sum up to 1
        # row vector a_j = a_j / sum(a_j)
        # add constant gamma to avoid 0-division
        gamma = tf.constant(1e-12, shape=(1, 1, 1), dtype='float32')
        sum_att = tf.reduce_sum(attention_weights, axis=-1) + gamma
        attention_weights = attention_weights / tf.expand_dims(sum_att,
                                                               axis=-1)

        return attention_weights

    def scaled_dot_product_attention(self, q, k, v, regularization, p,
                                     training):
        """
        Scaled-dot-product attention following Attention is all you need (2017)


        Args:
            q, k, v == (..., features, depth)
            regularization: 'dropout' or 'dropAttention', where 'dropout' does
                             not affect scaled_dot_product_attention
            p: dropattention probability
            training: True for training, False at inference

        Returns:
            scaled_values, attention weights

        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., features,
        #                                                features)
        # scale with sqrt(d_model)
        dk = tf.cast(tf.shape(k)[-1], dtype='float32')
        scaled_matmul_qk = matmul_qk / tf.math.sqrt(dk)
        # compute attention weights along last axis so that attention weights
        # add up to 1
        attention_weights = tf.nn.softmax(scaled_matmul_qk, axis=-1)
        # (..., features, features)
        # apply dropattention if desired
        if regularization == 'dropAttention' and training:
            attention_weights = self.dropAttention(attention_weights, p)

        # scale values with attention weights
        scaled_values = tf.matmul(attention_weights, v)
        # shape == (batch, num_heads, n_features, depth)

        return scaled_values, attention_weights

    def call(self, q, k, v, training):
        batch_size = tf.shape(q)[0]
        n_features = tf.shape(q)[1]

        # compute queries, keys and values linear transformation
        q = self.wq(q)  # (batch, features, d_model)
        k = self.wk(k)  # (batch, features, d_model)
        v = self.wv(v)  # (batch, features, d_model)
        # split heads
        q = self.split_heads(q, batch_size, n_features)
        # (batch, num_heads, features, depth)
        k = self.split_heads(k, batch_size, n_features)
        # (batch, num_heads, features, depth)
        v = self.split_heads(v, batch_size, n_features)
        # (batch, num_heads, features, depth)

        # compute attention
        # att_out == shape (batch_size, num_heads, depth)
        # att_weights == shape (batch_size, num_heads, features, features)

        att_out, att_weights = self.scaled_dot_product_attention(
                                    q, k, v,
                                    self.regularization, self.p, training)

        # concat attention outputs

        att_out = tf.transpose(att_out, perm=[0, 2, 1, 3])
        # shape == (batch, n_features, num_heads, depth)
        concat_att_out = tf.reshape(att_out, (batch_size,
                                    tf.shape(att_out)[1], self.d_model))
        # (batch, n_features, d_model)

        # apply linear projection to combine outputs of the different heads
        if self.num_heads > 1:
            concat_att_out = self.dense(concat_att_out)

        return concat_att_out, att_weights, v


class prediction_block(Layer):

    """ Block of a series of Conv1D

         Input shape:
            3D Tensor with shape:
             [batch, Tx, n features]
        Output shape:
            3D Tensor with shape:
            [batch, Tx, 1]
    """
    def __init__(self, filters=8, kernel_size=2,
                 activation=tf.keras.activations.swish,
                 dilation_rates=[1, 2], dropout=0.1, padding='causal',
                 kernel_initializer='he_uniform'):
        super(prediction_block, self).__init__()
        self.convs = []
        for dilation_rate in dilation_rates:
            self.convs.append(Conv1D(
                    filters, kernel_size, padding=padding,
                    data_format='channels_last', dilation_rate=dilation_rate,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.HeUniform()))
        self.out = Conv1D(
                    1, 1, padding='same', data_format='channels_last',
                    kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.dropout = Dropout(dropout)
        self.filters = filters

    def call(self, x):
        # Conv1D
        for conv in self.convs:
            x = conv(x)
            x = self.dropout(x)
        # 2D Convolution to merge channels
        out = self.out(x)

        return out


@keras_export('keras.layers.DepthwiseConv1D')
class DepthwiseConv1D(Conv1D):
    """Depthwise 1D convolution.
    Depthwise convolution consist of performing a depthwise convolution
    that acts separately on channels.
    Arguments:
        kernel_size: A single integer specifying the spatial
            dimensions of the filters.
        dilation_rate: A single integer specifying the dilation factor of the
                       convolution.
        strides: A single integer specifying the strides of the convolution.
            Specifying any `stride` value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"same"`, or `"causal"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input
            such that output has the same height/width dimension as the input.
            `"causal"` results in causal(dilated)
            convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
        depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        data_format: A string,
            one of "channels_last"(default) or "channels_first".
            [..., length, channels] or [..., channels, length]
        activation: Activation function to use.
            If nothing is specified, no activation is applied (a(x)=x)
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix.
        bias_initializer: Initializer for the bias vector.
        depthwise regularizer: Regularizer function applied to depthwise the
                               depthwise kernel matrix.
        bias_regularizer: Regularizer function applied to the bias vector
        activity_regularizer: Regularizer applied to the activation.
        depthwise_constraint: Constraint function applied to the depthwise
                              kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        3D Tensor with shape:
        `[batch, channels, length]` if data_format = "channels_first"
        or
        3D Tensor with shape:
        `[batch, length, channels]` if data_format = "channels_last"
    Output shape:
        3D Tensor with shape:
        `[batch, filters, new_length]` if data_format = "channels_first"
        or
        3D Tensor with shape:
        `[batch, new_length, filters]` if data_format = "channels_last"

    """

    def __init__(self,
                 kernel_size,
                 dilation_rate=1,
                 strides=1,
                 padding='causal',
                 depth_multiplier=1,
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv1D, self).__init__(
            filters=None,
            dilation_rate=dilation_rate,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError('Inputs to `DepthwiseConv1D` should have rank 3'
                             'Recieved input shape:', str(input_shape))
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the input to '
                             '`DepthwiseConv1D` should be defined.'
                             ' Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim*self.depth_multiplier,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # causal padding of inputs by left padding along the sequence axis
        if self.padding == 'causal':
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
        if self.data_format == 'channels_last':
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2

        # Explicitly broadcast inputs and kernels to 4D.
        inputs = array_ops.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
        dilation_rate = (1,) + self.dilation_rate

        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding

        # Compute depthwiseConv2D on broadcasted inputs
        outputs = nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=op_padding.upper(),
            dilations=dilation_rate,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       ndim=4))

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format,
                                                           ndim=4))

        outputs = array_ops.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            length = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            length = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        # length = conv_utils.conv_output_length(length, self.kernel_size,
        #                                        self.padding,
        #                                        self.strides)
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, length)
        elif self.data_format == 'channels_last':
            return (input_shape[0], length, out_filters)


# Interpretable NN class

class TFAN(tf.keras.Model):
    """
    Arguments:

    residual_blocks: integer specifying the amount of residual blocks
    residual_dropout: float, sepcifying the dropout rate in the residual
                      blocks.
    activation: Activation to use for depthwise convolutions and all final
                convolutions but the last (linear).
    depthwise_padding: padding style used for Depthwise 1D Convolutions:
                       one of "valid", "same", "causal"
    depthwise_kernel_size: Integer specifying depthwise kernel size.
    depth_multiplier: Integer, specifying the amount of filter per channel in
                      the depthwise convolutions.
    Tx: Integer specifying amount of time-steps per feature in input data.
    kernel_initializer: Weight initializer used in residual block and final
                        convolutions
    num_heads: Integer specifiyng the amount of heads in Multi-Head-Attention.
    d_model: Dimensionality of attention mechanism in Multi-Head-Attention.
    regularization: A string, being one of "dropout" or "dropAttention"
    p: A float, between [0,1] specifiying the regularization probability for
       the method specified in "regularization".
    final_filters: Integer specifying the amount of output channels in the
                   prediction block Conv1D.
    final_kernel_size: Integer specifying the kernel size of the 1d-Convolution
                       in the prediction block.
    final_dilations: List of dilations, where one Conv1D layer will be
                     initialised per entry in list.
    final_padding: padding style used for final 1D Convolutions: one of
                   "valid", "same", "causal"
    final_dropout: float, sepcifying the dropout rate in the prediction block.

    Input shape:
        x: 3D tensor of shape == (batch, Tx, n features)
    Outputs shape:
        out: 3D tensor of shape == (batch, Tx, 1) target time series
        att_weights: attention weights of MHA of shape == (batch, num_heads,
                                                           n features,
                                                           n features)
        values: values of MHA of shape == (batch, num_heads, n features,
                                           d_model(num_heads))

    """
    def __init__(self, residual_blocks=4, residual_dropout=0.2,
                 activation=tf.keras.activations.swish,
                 depthwise_padding="causal", depthwise_kernel_size=2,
                 depth_multiplier=1, Tx=20,
                 kernel_initializer=tf.keras.initializers.HeUniform(),
                 num_heads=8, d_model=32,
                 regularization="dropout", p=0.25,
                 final_filters=8, final_kernel_size=2,
                 final_dilations=[1, 2], final_padding='causal',
                 final_dropout=0.2):
        super(TFAN, self).__init__()

        # Residual Blocks
        self.res_blocks = []
        for dilation in range(residual_blocks):
            self.res_blocks.append(residual_block(
                 depthwise_kernel_size,
                 dilation_rate=dilation + 1,
                 padding=depthwise_padding, activation=activation,
                 dropout=residual_dropout,
                 depth_multiplier=depth_multiplier,
                 depthwise_initializer=kernel_initializer,
                 name='residual_block_{}'.format(dilation)))
        self.depth_multiplier = depth_multiplier
        # deal with tf bug -> see layer
        self.deal_with_tf_bug = deal_with_tf_bug(Tx)
        # Merging of depthwise channels
        merge_channels_trainable = (depth_multiplier > 1)
        self.merge_channels = DepthwiseConv1D(
                1, dilation_rate=1,
                data_format='channels_last', name='merge_channels',
                padding='same', activation='linear', depth_multiplier=1,
                depthwise_initializer='glorot_uniform', trainable=merge_channels_trainable)
        # MHA
        self.mha = MultiHeadAttention(d_model, num_heads, regularization, p)
        # Final Convolutions
        self.pred_block = prediction_block(
                    filters=final_filters,
                    kernel_size=final_kernel_size,
                    activation=activation, dilation_rates=final_dilations,
                    dropout=final_dropout, padding=final_padding,
                    kernel_initializer=kernel_initializer)

    def call(self, x, training=None):

        # residual blocks
        for block in self.res_blocks:
            x = block(x)
        # x of shape == (batch, Tx, n features*depth_multiplier)

        # merge channels if necessary
        if self.depth_multiplier > 1:
            x = self.merge_channels(x)
            x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1],
                               int(tf.shape(x)[-1] / self.depth_multiplier),
                               self.depth_multiplier))
            x = tf.reduce_sum(x, axis=-1)
        # x of shape == (batch, Tx, n features)

        # transpose to shape required by mha
        x = tf.transpose(x, perm=[0, 2, 1])
        # x of shape == (batch, n features, Tx)

        # apply dummy layer to deal with tensorflow bug
        # this layer will not impact x
        # it is just necessary for tf to catch up on the shapes, this will prevent a value error in mha
        x = self.deal_with_tf_bug(x)

        # MHA
        x, att, val = self.mha(x, x, x, training)

        # x of shape == (batch, n features, d_model)
        # att of shape == (batch, num_heads, n features, n features)
        # val of shape == (batch, num_heads, n features, depth)

        # transpose
        x = tf.transpose(x, perm=[0, 2, 1])
        # x of shape == (batch, Tx, n features)

        # Prediction
        out = self.pred_block(x)  # shape == (batch, Tx, 1)

        return out, att, val

        # override train step to only pass "out" to loss and ignore
        # attention weights and values

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # forward pass
            y_pred, _, _ = self(x, training=True)
            # Compute the loss
            loss = self.compiled_loss(y, y_pred,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
