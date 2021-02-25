import graph
import tensorflow as tf
import scipy.sparse
import numpy as np
import os


def cgcnn(input,label, L, L_2, F, K, M, regularization=0, reuse = None ):
    print('cgcnn network construction')
    regularizers = []
    # j = 0
    # tmp_L = []
    # for pp in p:
    #     tmp_L.append(L[j])
    #     j += int(np.log2(pp)) if pp > 1 else 0
    # L = tmp_L

    """Build the computational graph of the model."""
    with tf.variable_scope("EEG_GCN") as scope:
        if reuse:
            scope.reuse_variables()
            dropout = 1
        else:
            dropout = 0.5

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        A_matrix = scipy.sparse.csr_matrix(L[0])
        A_matrix = A_matrix.tocoo()
        indices = np.column_stack((A_matrix.row, A_matrix.col))
        A_2 = tf.SparseTensor(indices, A_matrix.data, A_matrix.shape)
        A_dense = tf.sparse_tensor_to_dense(A_2)
        # init = tf.constant_initializer(A_dense)

        """ Make the adjacency matrix learned"""
        adj = tf.get_variable('adj', initializer=A_dense)
        regularizers.append(tf.norm(adj, ord=1))
        adj_var = tf.nn.relu(adj)
        laplacian = graph.laplacian_tf(adj_var, normalized=True)
        laplacian = graph.rescale_tf_L(laplacian, lmax=2)
        laplacian = tf.nn.relu(laplacian)

        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        A_matrix_2 = scipy.sparse.csr_matrix(L_2[0])
        A_matrix_2 = A_matrix_2.tocoo()
        indices_2 = np.column_stack((A_matrix_2.row, A_matrix_2.col))
        A_22 = tf.SparseTensor(indices_2, A_matrix_2.data, A_matrix_2.shape)
        A_dense_2 = tf.sparse_tensor_to_dense(A_22)
        # init = tf.constant_initializer(A_dense)

        """ Make the adjacency matrix learned"""
        adj_2 = tf.get_variable('adj_2', initializer=A_dense_2)
        regularizers.append(tf.norm(adj_2, ord=1))
        adj_var_2 = tf.nn.relu(adj_2)
        laplacian_2 = graph.laplacian_tf(adj_var_2, normalized=True)
        laplacian_2 = graph.rescale_tf_L(laplacian_2, lmax=2)
        laplacian_2 = tf.nn.relu(laplacian_2)

        """input"""
        x_input = input  # (batch, channels , # of freq band) (64, 22, 26)
        _N, _M, _F = x_input.get_shape()

        """GCN for the freq domain"""
        x = tf.reshape(x_input, [-1, _F])  # 64*22 x 26
        x = tf.expand_dims(x, 2) #64*22 , 26, 1

        with tf.variable_scope('freq_conv{0}'.format(1)):
            with tf.name_scope('filter'):
                gf_x = chebyshev5(regularizers, x, laplacian_2, 2, 3)
                gf_x = batch_norm(gf_x)
            with tf.name_scope('bias_relu'):
                gf_x = b1relu(regularizers, gf_x)
        gf_x = tf.reshape(gf_x, [_N, _M, -1])

        """ GCN block 1"""
        with tf.variable_scope('conv{}'.format(1)):
            with tf.name_scope('filter'):
                g1_x = chebyshev5(regularizers, gf_x, laplacian, F[0], K[0]) # (batch, 24, # of kernel1)   #(batch, 24, # of kernel2)
                g1_x = batch_norm(g1_x)
            with tf.name_scope('bias_relu'):
                x = b1relu(regularizers, g1_x)

        """ GCN block 2"""
        with tf.variable_scope('conv{}'.format(2)):
            with tf.name_scope('filter'):
                g2_x = chebyshev5(regularizers, g1_x, laplacian, F[1], K[1])
                g2_x = batch_norm(g2_x)
            with tf.name_scope('bias_relu'):
                g2_x = b1relu(regularizers, g2_x)

        """ GCN block 3"""
        with tf.variable_scope('conv{}'.format(3)):
            with tf.name_scope('filter'):
                g2_x = chebyshev5(regularizers, g2_x, laplacian, F[2], K[2])
                g2_x = batch_norm(g2_x)
            with tf.name_scope('bias_relu'):
                g2_x = b1relu(regularizers, g2_x)

        """ 1*1 convolution"""
        with tf.variable_scope('11conv{0}'.format(2)):
            with tf.name_scope('11conv'):
                w2, b2 = get_params(name="11conv", shape=(1, F[2], 60), featurecnt=60)
                c2_x = _conv1d(g2_x, w2, b2)
                c2_x = batch_norm(c2_x)
                c2_x = b1relu(regularizers, c2_x)

        """Fully connected network"""
        _N, _M, _F = c2_x.get_shape() #(batch, 24, # of kernel2)
        x = tf.reshape(c2_x, [int(_N), int(_M * _F)])  # N x M
        for i, _M in enumerate(M[:-1]):
            with tf.variable_scope('fc{}'.format(i + 1)):
                x = fc(regularizers, x, _M)
                x = tf.nn.dropout(x, dropout)

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
        # loss = cross_entropy
    return logit, prediction, adj_2, loss # TODO: Haufe et al., 2014.


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

    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    W = _weight_variable(regularizers, [Fin * K, Fout], regularization=True)
    x = tf.matmul(x, W)  # N*M x Fout
    return tf.reshape(x, [N, M, Fout])  # N x M x Fout

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

def get_params(name, shape, featurecnt):
    w = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    b = tf.Variable(initial_value=tf.constant(0.1, shape=[featurecnt], dtype=tf.float32), name=name + "b")
    return w, b

def batch_norm(data):
    return tf.nn.batch_normalization(x=data, mean= 0, variance=1, offset=None, scale=None,variance_epsilon=1e-8)

# def batch_norm(input, is_training, name, momentum=0.9):
#     return tf.layers.batch_normalization(input, training=is_training, momentum=momentum, name=name, reuse=tf.AUTO_REUSE)

