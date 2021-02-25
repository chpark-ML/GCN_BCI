import graph
import tensorflow as tf
import scipy.sparse
import numpy as np
import os

def cgcnn(input,label, L, F, K, M, regularization=0, reuse = None , f_stride = 1, f_range = 3):
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
        # Compressed Sparse Row matrix
        A_matrix = scipy.sparse.csr_matrix(L[0])
        A_matrix = A_matrix.tocoo()
        indices = np.column_stack((A_matrix.row, A_matrix.col))
        A_sparse_tensor = tf.SparseTensor(indices, A_matrix.data, A_matrix.shape)
        A_dense = tf.sparse_tensor_to_dense(A_sparse_tensor)
        # init = tf.constant_initializer(A_dense)

        """input setting"""
        n_batch, n_ch, n_PSD = input.get_shape()  # (64, 22, 5)
        n_batch, n_ch, n_PSD= int(n_batch), int(n_ch), int(n_PSD)
        n_band = int(((n_PSD - f_range)/ f_stride)) + 1

        """spectral feature"""

        with tf.variable_scope('spectral_feature_conv1'):
            with tf.name_scope('conv1'):
                x = tf.expand_dims(input, 3)  # 1 x M x Fin*N
                n_feature1 = 10
                w1a = _weight_variable(regularizers, [1, 10, 1, n_feature1], regularization = True)
                b1a = _bias_variable(regularizers, [n_feature1], regularization=True)
                x= conv2d(input=x, weight=w1a, bias=b1a, padding='VALID')
                x= batch_norm(x)
                # x = tf.reshape(x, shape=(N_tmp, M_tmp, -1))

        with tf.variable_scope('spectral_feature_conv2'):
            with tf.name_scope('conv2'):
                # x = tf.expand_dims(input, 3)  # 1 x M x Fin*N
                # N_tmp, M_tmp, Fin = x.get_shape()  # (64, 22, 3)
                # N_tmp, M_tmp, Fin = int(N_tmp), int(M_tmp), int(Fin)
                n_feature2 = 20
                w1a = _weight_variable(regularizers, [1, 10, n_feature1, n_feature2], regularization=True)
                b1a = _bias_variable(regularizers, [n_feature2], regularization=True)
                # x = tf.expand_dims(x, 3)  # 1 x M x Fin*N
                x = conv2d(input=x, weight=w1a, bias=b1a, padding='VALID')
                x = batch_norm(x)
                # x = tf.reshape(x, shape=(N_tmp,M_tmp,-1))

        with tf.variable_scope('spectral_feature_conv3'):
            with tf.name_scope('conv3'):
                # x = tf.expand_dims(input, 3)  # 1 x M x Fin*N
                # N_tmp, M_tmp, Fin = x.get_shape()  # (64, 22, 3)
                # N_tmp, M_tmp, Fin = int(N_tmp), int(M_tmp), int(Fin)
                n_feature3 = 40
                w1a = _weight_variable(regularizers, [1, 10, n_feature2, n_feature3], regularization=True)
                b1a = _bias_variable(regularizers, [n_feature3], regularization=True)
                # x = tf.expand_dims(x, 3)  # 1 x M x Fin*N
                x = conv2d(input=x, weight=w1a, bias=b1a, padding='VALID')
                x = batch_norm(x)
                # x = tf.reshape(x, shape=(N_tmp,M_tmp,-1))

        with tf.variable_scope('spectral_feature_conv4'):
            with tf.name_scope('conv4'):
                # x = tf.expand_dims(input, 3)  # 1 x M x Fin*N
                N_tmp, M_tmp, n_psd, Fin = x.get_shape()  # (64, 22, 3)
                N_tmp, M_tmp, n_psd, Fin = int(N_tmp), int(M_tmp), int(n_psd), int(Fin)
                n_feature4 = 80
                w1a = _weight_variable(regularizers, [1, 10, n_feature3, n_feature4], regularization=True)
                b1a = _bias_variable(regularizers, [n_feature4], regularization=True)
                x = conv2d(input=x, weight=w1a, bias=b1a, padding='VALID')
                x = batch_norm(x)
                x = tf.reshape(x, shape=(N_tmp,M_tmp,-1))

        """ 1*1 convolution"""
        with tf.variable_scope('1_1conv_spec{0}'.format(1)):
            with tf.name_scope('11conv'):
                n_fout = 80
                w2 = _weight_variable(regularizers, [1, n_feature4, n_fout], regularization = True)
                b2 = _bias_variable(regularizers, [n_fout], regularization = True)
                x = _conv1d(x, w2, b2)
            with tf.name_scope('batchNorm'):
                x = batch_norm(x)
            with tf.variable_scope('bias_relu'):
                x = b1relu(regularizers, x)


        """band filtering"""
        for j in range(n_band):
            x = x[:,:,j*f_stride : j*f_stride + f_range]

            """adj matrix"""
            with tf.variable_scope('adj{}'.format(j)):
                adj = tf.get_variable('matrix', initializer=A_dense)
                # adj = A_dense
                adj = tf.nn.relu(adj)
                regularizers.append(tf.norm(adj, ord=1))
                laplacian = graph.laplacian_tf(adj, normalized=True)
                laplacian = graph.rescale_tf_L(laplacian, lmax=2)
                laplacian = tf.nn.relu(laplacian)

            """GCN filtering + 11conv"""
            for i in range (len(F)):
                """Chebyshev"""
                with tf.variable_scope('conv{0}_{1}'.format(j,i)):
                    with tf.name_scope('filter'):
                        N_tmp, M_tmp, Fin = x.get_shape()  # (64, 22, 3)
                        N_tmp, M_tmp, Fin = int(N_tmp), int(M_tmp), int(Fin)
                        W = _weight_variable(regularizers, [Fin * K[i], F[i]], regularization=True)
                        x = chebyshev5(regularizers, x, laplacian, F[i], K[i])
                        x = tf.matmul(x, W)  # N*M x Fout
                        x = tf.reshape(x, [N_tmp, M_tmp, F[i]])  # N x M x Fout
                        x = batch_norm(x)
                    with tf.name_scope('bias_relu'):
                        x = b1relu(regularizers, x)

                # """ 1*1 convolution"""
                # with tf.variable_scope('1_1conv{0}_{1}'.format(j,i)):
                #     with tf.name_scope('11conv'):
                #         n_fout = F[i]
                #         w2 = _weight_variable(regularizers, [1, F[i], n_fout], regularization = True)
                #         b2 = _bias_variable(regularizers, [n_fout], regularization = True)
                #         x = _conv1d(x, w2, b2)
                #         # w2, b2 = get_params(name="11conv", shape=(1, F[4], 140), featurecnt=140)
                #     with tf.name_scope('batchNorm'):
                #         x = batch_norm(x)
                #     with tf.variable_scope('bias_relu'):
                #         x = b1relu(regularizers, x)

            #concat
            """stack the filtered features"""
            if j == 0:
                x_0 = tf.expand_dims(x, 3)  # 1 x M x Fin*N
            else :
                x = tf.expand_dims(x, 3)
                x_0 = tf.concat([x_0, x], axis=3)


        N_t, C_t, F_t, B_t   = x_0.get_shape()
        x = tf.reshape(x_0, [N_t,C_t, F_t* B_t])  # M x Fin*N

        """ 1*1 convolution"""
        with tf.variable_scope('1_1conv{0}'.format(1)):
            with tf.name_scope('11conv'):
                n_fout = 512
                w2 = _weight_variable(regularizers, [1, F_t*B_t, n_fout], regularization = True)
                b2 = _bias_variable(regularizers, [n_fout], regularization = True)
                x = _conv1d(x, w2, b2)
            with tf.name_scope('batchNorm'):
                x = batch_norm(x)

            with tf.variable_scope('bias_relu'):
                x = b1relu(regularizers, x)

        with tf.variable_scope('spatio_feature_conv1'):
            with tf.name_scope('conv1'):
                n_feature = 1024
                w1a = _weight_variable(regularizers, [22, n_fout, n_feature], regularization = True)
                b1a = _bias_variable(regularizers, [n_feature], regularization=True)
                x = _conv1d(x, w1a, b1a)
                x = batch_norm(x)


        # x = tf.reduce_mean(x, axis=1, keep_dims=True)

        """Fully connected network"""
        _N, _M, _F = x.get_shape() #(batch, 24, # of kernel2)
        x = tf.reshape(x, [int(_N), int(_M * _F)])  # N x M
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

    return logit, prediction, adj, loss # TODO: Haufe et al., 2014.

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

