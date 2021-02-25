import graph
import network as nt
import utils as ut
import setting as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, time, collections, shutil
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy import sparse
import scipy.sparse
np.random.seed(1)

"""GPU connection"""
# import GPUtil
# devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
# os.environ["CUDA_VISIBLE_DEVICES"] = devices

"""Initialize strings for data save and load"""
file_train = 'train'
file_val = 'val'
file_test = 'test'

"""load dataset, psd_welch, and save"""
# train_data, train_label, val_data, val_label = ut.load_dataset(training=True)
# test_data, test_label = ut.load_dataset(training=False) #(54432, 22, 512, 1), (54432, )
# ut.save_processed_data(train_data, train_label, fileName= file_train)
# if st.n_train != 72:
#     ut.save_processed_data(val_data, val_label, fileName= file_val)
# ut.save_processed_data(test_data, test_label, fileName= file_test)

"""Load saved psd dataset"""
dat_train_psd = np.load(st.data_path +"/psd" +"/dat_" + file_train + "_" + str(st.sbj) + ".npy")
lbl_train = np.load(st.data_path+"/psd" + "/lbl_" + file_train + "_" + str(st.sbj) + ".npy")

dat_val_psd = np.load(st.data_path +"/psd"+ "/dat_" + file_val + "_" + str(st.sbj) + ".npy")
lbl_val = np.load(st.data_path +"/psd"+ "/lbl_" + file_val + "_" + str(st.sbj) + ".npy")

dat_test_psd = np.load(st.data_path +"/psd"+ "/dat_" + file_test + "_" + str(st.sbj) + ".npy")
lbl_test = np.load(st.data_path+"/psd" + "/lbl_" + file_test + "_" + str(st.sbj) + ".npy")

"""save the PSD data as a image"""
# for i in range(260):
#     print(i)
#     fig = plt.figure()
#     plt.imshow(np.asarray(dat_train_psd[189* i + 90, :, :]), vmax= 1.0, vmin=0.0)
#     plt.pcolor
#     plt.colorbar()
#     # plt.show()
#     fig.savefig('plot/'+str(i) + "_plot.png")
#     plt.close(fig)

"""Select the frequency interest range"""
dat_train_psd = dat_train_psd[:, :, st.min_freq: st.max_freq+1] # (trials, channels, features(st.max_freq - st.min_freq + 1)))
if st.n_train != 72:
    dat_val_psd = dat_val_psd[:, :, st.min_freq: st.max_freq+1]
dat_test_psd = dat_test_psd[:, :, st.min_freq: st.max_freq+1]

"""Graph construction"""
## define the coordinate for calculating the distance between nodes
coordinate_channel = np.array([[0,2], [-2,1], [-1,1],[0,1],[1,1],[2,1],[-3,0],[-2,0],[-1,0],[0,0],[1,0],[2,0],[3,0],[-2,-1],[-1,-1],[0,-1],[1,-1],[2,-1],[-1,-2],[0,-2],[1,-2],[0,-3]])
dist, idx = graph.distance_scipy_spatial(coordinate_channel, k=st.n_knn)
A = [graph.adjacency(dist, idx).astype(np.float32)]

train_tmp = dat_train_psd
train_tmp = np.transpose(train_tmp, (2, 1, 0))
train_tmp = np.reshape(train_tmp, (st.max_freq - st.min_freq + 1, -1))

## high level graph
dist_2, idx_2 = graph.distance_scipy_spatial(train_tmp, k=int((st.max_freq - st.min_freq + 1)/2),metric='correlation') ## node * feature
A_2 = [graph.adjacency(dist_2, idx_2).astype(np.float32)]

"""find the relation using illustration"""
# adj = graph.AdjFromIllust()
# A = [sparse.csr_matrix(adj)]

"""Set the params"""
params = dict()
# Architecture.
params['F']              = [80, 120, 160]  # Number of graph convolutional filters.
params['K']              = [3, 3, 3]  # Polynomial orders.
# params['p']              = [1]    # Pooling sizes.
params['M']              = [128,  st.n_class]  # Output dimensionality of fully connected layers.
# Optimization.
params['regularization'] = 5e-4
params['reuse']       = False



"""PlaceHolder"""
ph_X = tf.placeholder(dtype=tf.float32, shape=[st.batch_size, dat_train_psd.shape[1], dat_train_psd.shape[2]])
ph_Y = tf.placeholder(dtype=tf.float32, shape=[st.batch_size])
ph_X_test = tf.placeholder(dtype=tf.float32, shape=[(st.n_timepoint - st.test_win_size + 1), dat_test_psd.shape[1], dat_test_psd.shape[2]])
ph_Y_test = tf.placeholder(dtype=tf.float32, shape=[(st.n_timepoint - st.test_win_size + 1)])

"""global step"""
global_step = tf.Variable(0, trainable=False, name='global_step')

"""training model"""
logit, _ , adj_train, loss= nt.cgcnn(input=ph_X, label=ph_Y, L=A, L_2=A_2, **params)
variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="EEG_GCN")


"""test model"""
params["reuse"] = True
logit_test, pred, adj_test,loss_test= nt.cgcnn(input=ph_X_test,label=ph_Y_test, L=A, L_2=A_2, **params)

"""learning rate + optimizer"""
stepForEveryDecay = st.stepForEveryDecay
rateForDecay = st.rateForDecay
momentum = st.momentum
learning_rate = tf.train.exponential_decay(st.learning_rate, global_step, stepForEveryDecay, rateForDecay, staircase=True)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss=loss, var_list=variables, global_step=global_step)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss=loss, var_list=variables, global_step=global_step)

"""Session"""
print("subject num : " + str(st.sbj))
sess = tf.Session()
# saver = tf.train.Saver(tf.global_variables())
test_global_step = 0
sess.run(tf.global_variables_initializer())


"""log directory"""
log_dir = './logs/' + str(st.sbj)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = tf.summary.FileWriter(log_dir, sess.graph)

"""start train"""
indices = collections.deque()
num_steps = int(st.num_epochs * dat_train_psd.shape[0] / st.batch_size)
peak_test_kappa = 0
for step in range(1, num_steps + 1):
    # Be sure to have used all the samples before using one a second time.
    if len(indices) < st.batch_size:
        indices.extend(np.random.permutation(dat_train_psd.shape[0]))
    idx = [indices.popleft() for i in range(st.batch_size)]

    ## Set the batch data
    batch_data, batch_labels = dat_train_psd[idx, :, :], lbl_train[idx]
    if type(batch_data) is not np.ndarray:
        batch_data = batch_data.toarray()  # convert sparse matrices

    """training"""
    feed_dict = {ph_X: batch_data, ph_Y: batch_labels}
    # _, loss_, currentLR, adj_ = sess.run([optimizer, loss, learning_rate, adj_train], feed_dict=feed_dict)
    """summary"""
    _, loss_, currentLR, adj_ = sess.run([optimizer, loss, learning_rate, adj_train], feed_dict=feed_dict)
    summary = tf.Summary()
    summary.value.add(tag='train_loss', simple_value = loss_)
    writer.add_summary(summary, step)

    # Periodical evaluation of the model.
    if step % st.eval_frequency == 0 or step == num_steps:
        print("--- global_step : {:d} / {:d}---".format(sess.run(global_step), num_steps + 1))

        """validation"""
        if st.n_train != 72 :
            prediction = np.zeros(shape=(st.n_val_trial))
            ground_truth = np.zeros(shape=(st.n_val_trial))
            for trials in range(0, st.n_val_trial):
                test_global_step = test_global_step + 1
                test_batch_x = dat_val_psd[trials * st.test_n_rolled:(trials + 1) * st.test_n_rolled, :, :]
                test_batch_y = lbl_val[trials * st.test_n_rolled:(trials + 1) * st.test_n_rolled]
                pred_ = sess.run([pred], feed_dict={ph_X_test: test_batch_x})
                pred_ = np.argmax(np.bincount(np.squeeze(np.asarray(pred_)), minlength=st.n_class), axis=-1)
                ground_truth[trials] = lbl_val[trials * st.test_n_rolled]
                prediction[trials] = pred_
            ##  Accuracy
            # print("validation Accuracy: %f, Kappa value: %f"
            #       % (accuracy_score(y_true=ground_truth, y_pred=prediction),
            #          cohen_kappa_score(y1=ground_truth, y2=prediction)))
            """summary"""
            # summary = tf.Summary()
            summary.value.add(tag='val_kappa', simple_value=cohen_kappa_score(y1=ground_truth, y2=prediction))
            writer.add_summary(summary, step)
        """test"""
        prediction = np.zeros(shape=(st.n_test_trial))
        ground_truth = np.zeros(shape=(st.n_test_trial))
        for trials in range(0, st.n_test_trial):
            test_global_step = test_global_step + 1
            test_batch_x = dat_test_psd[trials * st.test_n_rolled:(trials + 1) * st.test_n_rolled, :, :]
            test_batch_y = lbl_test[trials * st.test_n_rolled:(trials + 1) * st.test_n_rolled]
            pred_, adj__ = sess.run([pred, adj_test], feed_dict={ph_X_test: test_batch_x})
            pred_ = np.argmax(np.bincount(np.squeeze(np.asarray(pred_)), minlength=st.n_class), axis=-1)
            ground_truth[trials] = lbl_test[trials * st.test_n_rolled]
            prediction[trials] = pred_

        """Accuracy"""
        accuracy = accuracy_score(y_true=ground_truth, y_pred=prediction)
        kappa = cohen_kappa_score(y1=ground_truth, y2=prediction)
        """summary"""
        # summary = tf.Summary()
        summary.value.add(tag='test_kappa', simple_value=kappa)
        writer.add_summary(summary, step)


        if peak_test_kappa < kappa:
            peak_test_kappa = kappa

        print("test Accuracy: %f, Kappa value: %f" % (accuracy, kappa))
        print("peak Kappa : {0}".format(peak_test_kappa))

        # TODO : show the adjacency matrix when the test accuracy is measured
        # plt.imshow(np.asarray(adj__))
        # plt.pcolor
        # plt.colorbar()
        # plt.show()
        # print("step : {0}".format(step))
writer.close()
sess.close()
