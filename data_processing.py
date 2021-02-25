
import utils as ut
import setting as st
import numpy as np

np.random.seed(1)
## initialize strings for data save and load
file_train = 'train'
file_val = 'val'
file_test = 'test'

for sbj in range (1, 10):
    print(sbj)
    train_data, train_label, val_data, val_label = ut.load_dataset(training=True, sbj= sbj)
    test_data, test_label = ut.load_dataset(training=False, sbj= sbj) #(54432, 22, 512, 1), (54432, )

    ut.save_processed_data(train_data, train_label, fileName= file_train, sbj = sbj, n_per_seg=None, n_overlap=2)
    if st.n_train != 72:
        ut.save_processed_data(val_data, val_label, fileName= file_val, sbj = sbj)
    ut.save_processed_data(test_data, test_label, fileName= file_test, sbj = sbj)

