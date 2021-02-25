import utils as ut
import setting as st
import numpy as np
import matplotlib.pyplot as plt
import os
from mne.time_frequency import psd_array_welch
## initialize strings for data save and load

"""subject"""
for sbj in range(1, 10):
    print("sbj : " + str(sbj))
    train_data = ut.load_dataset_label(sbj=sbj)  # (4, 22, 700, 72) #(54432, 22, 512, 1),
    train_data = np.swapaxes(train_data, axis1=0, axis2=3)  ##(72, 22, 700, 4)

    """class"""
    for i in range(4):
        ## Create directroy
        dirName = 'psd_sbj_lbl/' + str(sbj) + " sbject" + "/" + str(i+1) + " class" + "/"

        if not os.path.exists(dirName):
            os.makedirs(dirName)

        psds = ut.proc_psd_welch(np.expand_dims(train_data[:,:,:,i], axis=3), sfreq=250, fmin=0, fmax=100, n_fft=250, n_per_seg=250, n_overlap=150)

        """print mean"""
        m_psds = np.mean(psds, axis=0)
        fig = plt.figure()
        plt.imshow(np.asarray(m_psds), vmax= 1.5, vmin=0.0)
        plt.pcolor
        plt.colorbar()
        fig.savefig(dirName + "mean.png")
        plt.close(fig)

        """trial"""
        for j in range(72):
            fig = plt.figure()
            plt.imshow(np.asarray(psds[j,:,:]), vmax= 1.5, vmin=0.0)
            plt.pcolor
            plt.colorbar()
            fig.savefig(dirName + str(j) + "_trial.png")
            plt.close(fig)