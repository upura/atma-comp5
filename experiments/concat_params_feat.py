from ayniy.utils import Data


fe005_tr = Data.load('../input/X_train_fe005.pkl')
fe005_te = Data.load('../input/X_test_fe005.pkl')

fe005_top100_tr = Data.load('../input/X_train_fe005_top100.pkl')
fe005_top100_te = Data.load('../input/X_test_fe005_top100.pkl')

fe005_top100_tr['abs_params2_minus_params5'] = (fe005_tr['params2'] - fe005_tr['params5']).abs()
fe005_top100_te['abs_params2_minus_params5'] = (fe005_te['params2'] / fe005_te['params5']).abs()

fe005_top100_tr['params1_div_params4'] = fe005_tr['params1'] / fe005_tr['params4']
fe005_top100_te['params1_div_params4'] = fe005_te['params1'] / fe005_te['params4']

fe005_top100_tr['params1_div_params3'] = fe005_tr['params1'] / fe005_tr['params3']
fe005_top100_te['params1_div_params3'] = fe005_te['params1'] / fe005_te['params3']

fe_name = 'fe005_top100_add_params'
Data.dump(fe005_top100_tr, f'../input/X_train_{fe_name}.pkl')
Data.dump(fe005_top100_te, f'../input/X_test_{fe_name}.pkl')
