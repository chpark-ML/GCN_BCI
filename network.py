import graph
import tensorflow as tf
import scipy.sparse
import numpy as np
import os

def cgcnn(input,label, L, F, K, M, regularization=0, reuse = None ):
    print('cgcnn network construction')
    regularizers = []
    with tf.variable_scope("EEG_GCN") as scope:
        if reuse:
            scope.reuse_variables()
            dropout_ = 1.0
        else:
            dropout_ = 0.5

        """The First Block of the RCL"""
        feature_size = 64
        nb_feature1 = feature_size / 4

        """1"""
        with tf.variable_scope('block1_1'):
            w1a = _weight_variable(regularizers, [1,9,1, nb_feature1], regularization=False)
            b1a = _bias_variable(regularizers, shape=(nb_feature1), regularization=False)
            """convolution"""
            conv1a = conv2d(input=input, weight=w1a, bias=b1a, padding="SAME")
            conv1a = batch_norm(conv1a)

        """2"""
        with tf.variable_scope('block1_2'):
            w1b = _weight_variable(regularizers, [1, 9, nb_feature1, nb_feature1], regularization=False)
            b1b = _bias_variable(regularizers, shape=(nb_feature1), regularization=False)
            conv1b = conv2d(input=conv1a, weight=w1b, bias=b1b, padding="SAME")
            conv1b = conv1a + conv1b

        """3"""
        with tf.variable_scope('block1_3'):
            w1c = _weight_variable(regularizers, [1, 9, nb_feature1, nb_feature1], regularization=False)
            b1c = _bias_variable(regularizers, shape=(nb_feature1), regularization=False)
            conv1c = conv2d(input=conv1b, weight=w1c, bias=b1c, padding="SAME")
            conv1c = conv1a + conv1c

        conv1c = batch_norm(conv1c)
        conv1c = pool_layer(conv1c)
        conv1c = dropout(conv1c, dropout_)

        """The Second block of the RCL"""
        nb_feature2 = feature_size / 2
        """1"""
        w2, b2 = get_params(name="conv2", shape=(1, 1, nb_feature1, nb_feature2), featurecnt=nb_feature2)
        conv2 = conv2d(input=conv1c, weight=w2, bias=b2, padding="SAME")
        conv2 = batch_norm(conv2)

        """2"""
        w2a, b2a = get_params(name="conv2a", shape=(1, 9, nb_feature2, nb_feature2), featurecnt=nb_feature2)
        conv2a = conv2d(input=conv2, weight=w2a, bias=b2a, padding="SAME")
        conv2a = conv2 + conv2a

        """3"""
        w2b, b2b = get_params(name="conv2b", shape=(1, 9, nb_feature2, nb_feature2), featurecnt=nb_feature2)
        conv2b = conv2d(input=conv2a, weight=w2b, bias=b2b, padding="SAME")
        conv2b = conv2 + conv2b

        """4"""
        w2c, b2c = get_params(name="conv2c", shape=(1, 9, nb_feature2, nb_feature2), featurecnt=nb_feature2)
        conv2c = conv2d(input=conv2b, weight=w2c, bias=b2c, padding="SAME")
        # conv2c = conv2 + conv2c
        conv2c = batch_norm(conv2c)
        conv2c = pool_layer(conv2c)
        conv2c = dropout(conv2c, dropout_)


        """The Third block of the RCL"""
        nb_feature3 = feature_size

        """1"""
        w3, b3 = get_params(name="conv3", shape=(1, 1, nb_feature2, nb_feature3), featurecnt=nb_feature3)
        conv3 = conv2d(input=conv2c, weight=w3, bias=b3, padding="SAME")
        conv3 = batch_norm(conv3)

        """2"""
        w3a, b3a = get_params(name="conv3a", shape=(1, 9, nb_feature3, nb_feature3), featurecnt=nb_feature3)
        conv3a = conv2d(input=conv3, weight=w3a, bias=b3a, padding="SAME")
        conv3a = conv3 + conv3a

        """3"""
        w3b, b3b = get_params(name="conv3b", shape=(1, 9, nb_feature3, nb_feature3), featurecnt=nb_feature3)
        conv3b = conv2d(input=conv3a, weight=w3b, bias=b3b, padding="SAME")
        conv3b = conv3 + conv3b

        """4"""
        w3c, b3c = get_params(name="conv3c", shape=(1, 9, nb_feature3, nb_feature3), featurecnt=nb_feature3)
        conv3c = conv2d(input=conv3b, weight=w3c, bias=b3c, padding="SAME")
        # conv3c = conv3 + conv3c
        conv3c = batch_norm(conv3c)  # (batch, 22, 8, 64)
        conv3c = pool_layer(conv3c)
        conv3c = dropout(conv3c, dropout_)

        # list_x = conv3c.get_shape().as_list()
        # x = tf.reshape(conv3c, [list_x[0] , list_x[1], -1])

        # """Rescale Laplacian and store as a TF sparse tensor"""
        # A_matrix = scipy.sparse.csr_matrix(L[0])
        # A_matrix = A_matrix.tocoo()
        # indices = np.column_stack((A_matrix.row, A_matrix.col))
        # A_sparse_tensor = tf.SparseTensor(indices, A_matrix.data, A_matrix.shape)
        # A_dense = tf.sparse_tensor_to_dense(A_sparse_tensor)
        #
        # """Variable adj matrix"""
        # with tf.variable_scope('adj'):
        #     adj = tf.get_variable('matrix', initializer=A_dense)
        #     # adj = A_dense
        #     adj = tf.nn.relu(adj)
        #     regularizers.append(tf.norm(adj, ord=1))
        #     laplacian = graph.laplacian_tf(adj, normalized=True)
        #     laplacian = graph.rescale_tf_L(laplacian, lmax=2)
        #     laplacian = tf.nn.relu(laplacian)
        #
        # """GCN filtering + 11conv"""
        # for i in range (len(F)):
        #     """Chebyshev"""
        #     with tf.variable_scope('conv{0}'.format(i)):
        #         with tf.name_scope('filter'):
        #             N_tmp, M_tmp, Fin = x.get_shape()  # (64, 22,512)
        #             N_tmp, M_tmp, Fin = int(N_tmp), int(M_tmp), int(Fin)
        #             W = _weight_variable(regularizers, [Fin * K[i], F[i]], regularization=True)
        #             x = chebyshev5(regularizers, x, laplacian, F[i], K[i])
        #             x = tf.matmul(x, W)  # N*M x Fout
        #             x = tf.reshape(x, [N_tmp, M_tmp, F[i]])  # N x M x Fout
        #             x = batch_norm(x)
        #         with tf.name_scope('bias_relu'):
        #             x = b1relu(regularizers, x)

        """spatio feature extraction"""
        n_feature = 64
        with tf.variable_scope('spatial_feature_extraction'):
            w1a = _weight_variable(regularizers, [22, 1, n_feature, n_feature], regularization=False)
            b1a = _bias_variable(regularizers, [n_feature], regularization=False)
            x = conv2d(input=conv3c, weight=w1a, bias=b1a, padding="VALID")
            x = batch_norm(x)
            """"""
            w3, b3 = get_params(name="11conv", shape=(1, 1, n_feature, n_feature), featurecnt=n_feature)
            x = conv2d(input=x, weight=w3, bias=b3, padding="SAME")
            x = batch_norm(x)

        """Fully connected network"""
        _N, _C, _T, _F = x.get_shape() #(batch, 24, # of kernel2)
        x = tf.reshape(x, [int(_N), int(_C *_T * _F)])  # N x M
        for i, _M in enumerate(M[:-1]):
            with tf.variable_scope('fc{}'.format(i + 1)):
                x = fc(regularizers, x, _M)
                x = tf.nn.dropout(x, dropout_)

        """Logits linear layer, i.e. softmax without normalization"""
        with tf.variable_scope('logits'):
            logit = fc(regularizers, x, M[-1], relu=False)
            prediction = tf.argmax(input=tf.nn.softmax(logit), axis=-1)

        """Calculation of the loss with the regularization"""
        with tf.variable_scope('loss'):
            label = tf.one_hot(indices=tf.cast(label, dtype=tf.int64), depth=4)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))
            regularization *= tf.add_n(regularizers)
        loss = cross_entropy + regularization

    return logit, prediction, loss  # TODO: Haufe et al., 2014.


def chebyshev5(regularizers, x, L, Fout, K):
    N, M, Fin = x.get_shape() #(64, 22, 5)
    N, M, Fin = int(N), int(M), int(Fin)

    # Transform to Chebyshev basis
    x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
    x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
    x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
        return tf.concat([x, x_], axis=0)  # K x M x Fin*N
    if K > 1:
        # x1 = tf.sparse_tensor_dense_matmul(L, x0)
        x1 = tf.matmul(L, x0)
        x = concat(x, x1)

    for k in range(2, K):
        # x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
        x2 = 2 * tf.matmul(L, x1) - x0

        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
    x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
    return x


def b1relu( regularizers, x):
    """Bias and ReLU. One bias per filter."""
    N, M, F = x.get_shape()
    b = _bias_variable(regularizers,[1, 1, int(F)], regularization=True)
    return tf.nn.relu(x + b)

def b2relu(regularizers, x ):
    """Bias and ReLU. One bias per vertex per filter."""
    N, M, F = x.get_shape()
    b = _bias_variable(regularizers,[1, int(M), int(F)], regularization=True)
    return tf.nn.relu(x + b)

def mpool1( x, p):
    """Max pooling of size p. Should be a power of 2."""
    if p > 1:
        x = tf.expand_dims(x, 3)  # N x M x F x 1
        x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
        # tf.maximum
        return tf.squeeze(x, [3])  # N x M/p x F
    else:
        return x

def apool1(x, p):
    """Average pooling of size p. Should be a power of 2."""
    if p > 1:
        x = tf.expand_dims(x, 3)  # N x M x F x 1
        x = tf.nn.avg_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
        return tf.squeeze(x, [3])  # N x M/p x F
    else:
        return x

def fc(regularizers, x, Mout, relu=True):
    """Fully connected layer with Mout features."""
    N, Min = x.get_shape()
    W = _weight_variable(regularizers,[int(Min), Mout], regularization=True)
    b = _bias_variable(regularizers,[Mout], regularization=True)
    x = tf.matmul(x, W) + b
    return tf.nn.relu(x) if relu else x


def _weight_variable(regularizers, shape, regularization=True):
    initial = tf.truncated_normal_initializer(0, 0.1)
    # initial = tf.contrib.layers.xavier_initializer()
    # initial = tf.contrib.layers.variance_scaling_initializer()
    var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
    if regularization:
        # regularizers.append(tf.nn.l2_loss(var))
        regularizers.append(tf.norm(var,ord=1))
    tf.summary.histogram(var.op.name, var)
    return var

def _bias_variable(regularizers, shape, regularization=True):
    initial = tf.constant_initializer(0.1)
    var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
    if regularization:
        # regularizers.append(tf.nn.l2_loss(var))
        regularizers.append(tf.norm(var, ord=1))
    tf.summary.histogram(var.op.name, var)
    return var


def _conv1d(x, W, b):
    return tf.nn.relu( tf.nn.bias_add (tf.nn.conv1d(x, W, stride=1, padding='VALID'), b))

def conv2d(input, weight, bias, padding, strides=[1, 1, 1, 1], activation_fcn="relu"):
    conv = tf.nn.conv2d(input=input, filter=weight, strides=strides, padding=padding)
    if activation_fcn == "leaky_relu":
        return leaky_relu(tf.nn.bias_add(conv, bias))
    else:
        return tf.nn.relu(tf.nn.bias_add(conv, bias))

def leaky_relu(x):
    return tf.maximum(0.2 * x, x)


def batch_norm(data):
    return tf.nn.batch_normalization(x=data, mean= 0, variance=1, offset=None, scale=None,variance_epsilon=1e-8)

# def batch_norm(input, is_training, name,momentum=0.9):
#     return tf.layers.batch_normalization(input, training=is_training, momentum=momentum,name=name,  reuse=tf.AUTO_REUSE)

def get_params(name, shape, featurecnt):
    w = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    b = tf.Variable(initial_value=tf.constant(0.1, shape=[featurecnt], dtype=tf.float32), name=name + "b")
    return w, b

def leaky_relu(x):
    return tf.maximum(0.2 * x, x)

def conv2d(input, weight, bias, padding, strides=[1, 1, 1, 1], activation_fcn="relu"):
    conv = tf.nn.conv2d(input=input, filter=weight, strides=strides, padding=padding)
    if activation_fcn == "leaky_relu":
        return leaky_relu(tf.nn.bias_add(conv, bias))
    else:
        return tf.nn.relu(tf.nn.bias_add(conv, bias))

def batch_norm(data):
    return tf.nn.batch_normalization(x=data, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-8)

def pool_layer(data):
    return tf.nn.max_pool(value=data, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding="VALID")

def dropout(data, dropout):
    return tf.nn.dropout(data, dropout)

def fclayer(data, name, num_hidden, activation_fn="relu"):
    num_input = data.get_shape().as_list()[-1]
    wfc, bfc = get_params(name=name, shape=(num_input, num_hidden), featurecnt=num_hidden)
    if activation_fn == "relu":
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(data, wfc), bfc))
    else:
        return tf.nn.bias_add(tf.matmul(data, wfc), bfc)
