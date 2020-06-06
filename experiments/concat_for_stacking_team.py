import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from ayniy.utils import Data


def load_oof_from_run_id(run_id: str):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    if run_id in ('run013', 'run014', 'run015'):
        oof = oof.reshape(-1, )
    return oof


def load_pred_from_run_id(run_id: str):
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if run_id in ('run013', 'run014', 'run015'):
        pred = pred.reshape(-1, )
    return pred


run_ids = [
    'run070',
    'run069',
    'run068',
    'run067',
    'run066',
    'run062',
    'run062',
    'run061',
    'run060',
    'run059',
    'run058',
    'run057',
    'run056',
    'run055',
    'run054',
    'run053',
    'run052',
    'run051',
    'run050',
    'run049',
    'run048',
    'run048',
    'run048',
    'run047',
    'run046',
    'run045',
    'run043',
    'run042',
    'run041',
    'run040',
    'run022',
    'run039',
    'run036',
    'run035',
    'run029',
    'run028',
    'run027',
    'run026',
    'run025',
    'run024',
    'run023',
    'run022',
    'run021',
    'run020',
    'run019',
    'run018',
    'run017',
    'run016',
    'run015',
    'run014',
    'run013',
    'run012',
    'run011',
    'run010',
    'run009',
    'run008',
    'run001',
]
fe_name = 'stack015'

train = pd.read_csv('../input/train.csv')
y_train = train['target']

# u++
oofs = [load_oof_from_run_id(ri) for ri in run_ids]
preds = [load_pred_from_run_id(ri) for ri in run_ids]

# kmat
dirname = 'result_0603_01_upf0'
kmat_preds = []
kmat_oofs = []
for i in range(5):
    kmat_preds.append(pd.read_pickle(f'../input/kmat/{dirname}/test_fold{i}.pickle'))
    kmat_oofs.append(pd.read_pickle(f'../input/kmat/{dirname}/val_fold{i}.pickle'))
kmat_pred = np.mean(preds, axis=0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
train['oof'] = np.nan
for i, (tr_idx, val_idx) in enumerate(cv.split(train, train['target'])):
    train.loc[val_idx, 'oof'] = kmat_oofs[i]
    # print(average_precision_score(
    #     train.loc[val_idx, 'target'],
    #     train.loc[val_idx, 'oof']))

preds.append(kmat_pred)
oofs.append(train['oof'])

for oof in oofs:
    print(average_precision_score(y_train, oof))

X_train = pd.DataFrame(np.stack(oofs).T)
X_test = pd.DataFrame(np.stack(preds).T)
X_train.columns = run_ids + ['kmat']
X_test.columns = run_ids + ['kmat']

Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')
