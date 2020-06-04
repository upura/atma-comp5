import pandas as pd
import numpy as np

from ayniy.utils import Data


# kmat
dirname = 'result_0604_03_upf0_features'
kmat_trains = []
kmat_tests = []
for i in range(5):
    kmat_trains.append(pd.read_pickle(f'../input/kmat/{dirname}/train_features_fold{i}.pickle'))
    kmat_tests.append(pd.read_pickle(f'../input/kmat/{dirname}/test_features_fold{i}.pickle'))
kmat_train = np.mean(kmat_trains, axis=0)
kmat_test = np.mean(kmat_tests, axis=0)

print(kmat_train.shape)
print(kmat_test.shape)

kmat_train = pd.DataFrame(kmat_train)
kmat_test = pd.DataFrame(kmat_test)
kmat_train.columns = [f'1dcnn{i}' for i in range(kmat_train.shape[1])]
kmat_test.columns = [f'1dcnn{i}' for i in range(kmat_train.shape[1])]

fe005_top100_tr = Data.load('../input/X_train_fe005_top100.pkl')
fe005_top100_te = Data.load('../input/X_test_fe005_top100.pkl')

train_fitting_ef_add = pd.concat([fe005_top100_tr, kmat_train], axis=1)
test_fitting_ef_add = pd.concat([fe005_top100_te, kmat_test], axis=1)

fe_name = 'fe005_top100_cnn'
Data.dump(train_fitting_ef_add, f'../input/X_train_{fe_name}.pkl')
Data.dump(test_fitting_ef_add, f'../input/X_test_{fe_name}.pkl')
