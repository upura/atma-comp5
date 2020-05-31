import pandas as pd

from ayniy.utils import Data


fe_id = 'fe004'
run_id = 'run022'
fe_name = 'fe004_top500'
N_FEATURES = 500

X_train = Data.load(f'../input/X_train_{fe_id}.pkl')
y_train = Data.load(f'../input/y_train_{fe_id}.pkl')
X_test = Data.load(f'../input/X_test_{fe_id}.pkl')

fi = pd.read_csv(f'../output/importance/{run_id}-fi.csv')['Feature'][:N_FEATURES]

X_train = X_train[fi]
X_test = X_test[fi]

Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')