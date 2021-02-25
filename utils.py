import scipy.sparse
import numpy as np
import tensorflow as tf
import scipy.io
import setting as st
from mne.time_frequency import psd_array_welch

def rolling_window(a, window):
    def rolling_window_lastaxis(a, window):
        if window < 1:
            raise ValueError("`window` must be at least 1.")
        if window > a.shape[-1]:
            raise ValueError("`window` is too long.")
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a

def load_dataset(training, sbj = st.sbj):
    path = st.data_path
    if training == True:
        n_train = st.n_train
        s_window = st.train_win_size
        data = np.array([scipy.io.loadmat(path + "/A%02dTClass1.mat" % sbj)["Class1"],
                scipy.io.loadmat(path + "/A%02dTClass2.mat" % sbj)["Class2"],
                scipy.io.loadmat(path + "/A%02dTClass3.mat" % sbj)["Class3"],
                scipy.io.loadmat(path + "/A%02dTClass4.mat" % sbj)["Class4"]]) #(4, 23, 700, 72)
                # (Each class, 22 channels + label, timepoints, trials)

        n_class, n_channel, n_timePoint, n_trial = data.shape
        n_channel = n_channel-1

        ## TODO : shuffle the data trial order
        indx = np.random.permutation(data.shape[-1])
        data = data[:, :, :, indx]

        t_dat_tmp = data[:,:,:,:n_train] ## (4,23,700,65)
        v_dat_tmp = data[:,:,:,n_train:] ## (4,23,700,7)



        t_dat = np.empty(shape=(0, n_channel, s_window, 1), dtype=np.float32)
        t_lbl = np.empty(shape=0, dtype=np.uint8)
        v_dat = np.empty(shape=(0, n_channel, s_window, 1), dtype=np.float32)
        v_lbl = np.empty(shape=0, dtype=np.uint8)

        for cnt, cur_dat in enumerate(data):
            ## train data rolling
            cur_t_dat = t_dat_tmp[cnt, :, :, :]
            cur_t_dat = np.swapaxes(cur_t_dat, 0, 2)[..., :-1]
            rolled_t_dat = rolling_window(cur_t_dat, (1, s_window))
            rolled_t_dat = rolled_t_dat.reshape(-1, n_channel, s_window)[..., None]
            t_dat = np.concatenate((t_dat, rolled_t_dat), axis=0)
            t_lbl = np.concatenate((t_lbl, np.full(shape=rolled_t_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0)

            ## val data rolling
            cur_v_dat = v_dat_tmp[cnt, :,:,:]
            cur_v_dat = np.swapaxes(cur_v_dat, 0, 2)[..., :-1]
            rolled_v_dat = rolling_window(cur_v_dat, (1, s_window))
            rolled_v_dat = rolled_v_dat.reshape(-1, n_channel, s_window)[..., None]
            v_dat = np.concatenate((v_dat, rolled_v_dat), axis=0)
            v_lbl = np.concatenate((v_lbl, np.full(shape=rolled_v_dat.shape[0], fill_value=cnt, dtype=np.uint8)), axis=0)

        return t_dat, t_lbl, v_dat, v_lbl #(54432, 22, 512, 1), (54432,)
    else:
        s_window = st.test_win_size
        data = np.array(scipy.io.loadmat(path + "/A%02dE" % sbj)["data"]) # (23, 700, 288)
        n_channel, n_timePoint, n_trial = data.shape

        temp = np.zeros(shape=((n_timePoint - s_window + 1)*n_trial, n_channel, s_window, 1))
        n_rolled = n_timePoint - s_window + 1
        for i in range(0, data.shape[-1]): #(trail)
            for j in range(0, n_rolled): #(number of the rolled window )
                temp[n_rolled * i + j, :, :, 0] = data[:, j:j + s_window, i]
        data = temp[:, :-1, :, :]  # (54432, 22, 512, 1)
        label = temp[:, -1, 0, 0] - 1  # (54432,)
        return data, label

def proc_psd_welch(dat, fmin, fmax, sfreq=250, n_fft=250, n_per_seg = None, n_overlap=0):
    dat = np.array(dat[:, :, :, 0])
    dat = np.swapaxes(dat, 0, 1)
    psds, freqs = psd_array_welch(dat, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft,  n_per_seg=n_per_seg, n_overlap=n_overlap)
    psds = np.swapaxes(psds, 0, 1)
    return psds

def calculate_loss(logit, label):
    label = tf.one_hot(indices=tf.cast(label, dtype=tf.int64), depth=4)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))

    return loss

def randomize_dataset(data, label):
    tmp_dat = np.zeros(shape=data.shape)
    tmp_lbl = np.zeros(shape=label.shape)
    rand_idx = np.random.permutation(data.shape[0])
    for idx in range(rand_idx.shape[0]):
        tmp_dat[idx, :, :, :] = data[rand_idx[idx], :, :, :]
        tmp_lbl[idx] = label[rand_idx[idx]]
    return tmp_dat, tmp_lbl

def sigmoid(x):
    return 1/(1+np.exp(-x))

def save_processed_data(input_dat, input_lbl, minFreq = 1, maxFreq =50, fileName = 'train', sbj = st.sbj, n_per_seg=None , n_overlap=0):
    # Call the psd_welch function to convert time-domain to frequency-domain
    psds = proc_psd_welch(dat=input_dat, fmin=minFreq, fmax=maxFreq, sfreq=250, n_fft=250, n_per_seg=n_per_seg , n_overlap=n_overlap)

    # Save the preprocessing the data
    np.save(st.data_path + "/psd" + "/dat_" + fileName + "_" + str(sbj), psds) ## dat_train_1
    np.save(st.data_path + "/psd"+"/lbl_" + fileName + "_" + str(sbj), input_lbl) ## lbl_train_1

def load_dataset_label(sbj=st.sbj):
    path = st.data_path
    n_train = st.n_train
    s_window = st.train_win_size
    data = np.array([scipy.io.loadmat(path + "/A%02dTClass1.mat" % sbj)["Class1"],
                     scipy.io.loadmat(path + "/A%02dTClass2.mat" % sbj)["Class2"],
                     scipy.io.loadmat(path + "/A%02dTClass3.mat" % sbj)["Class3"],
                     scipy.io.loadmat(path + "/A%02dTClass4.mat" % sbj)["Class4"]])  # (4, 23, 700, 72)
    # (Each class, 22 channels + label, timepoints, trials)
    return data[:,:-1,:,:] #(4, 22, 700, 72)