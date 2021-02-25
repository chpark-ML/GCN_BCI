## data_path
data_path = "D:/BCI_Competition_IV_IIa/separated_class"

## gpu4
# data_path = '/home/chpark/Data/BCIC_IV_IIA'

## setting
sbj = 5
n_train = 65 ##65
eval_frequency = 100

## parameters
batch_size = 64
num_epochs = 50
learning_rate = 1e-3
stepForEveryDecay = 500
rateForDecay = 0.97
momentum = 0.9

## graph
n_knn = 10

## fixed
n_class = 4
n_channel = 22
n_timepoint = 700
n_val_trial = (72 - n_train) * n_class
n_test_trial = 288

## rolling window size
train_win_size = 512
test_win_size = 512
test_n_rolled = n_timepoint - test_win_size + 1


## vector_feature
max_freq = 35
min_freq = 5
