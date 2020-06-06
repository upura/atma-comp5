import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from ayniy.utils import Data


def load_oof_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    if run_id in ('run091', 'run092', 'run097'):
        oof = oof.reshape(-1, )
    if to_rank:
        oof = rankdata(oof) / len(oof)
    return oof


def load_pred_from_run_id(run_id: str, to_rank: False):
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if run_id in ('run091', 'run092', 'run097'):
        pred = pred.reshape(-1, )
    if to_rank:
        pred = rankdata(pred) / len(pred)
    return pred


run_ids = [
    'run102',
    'run090',
    'run089',
    'run087',
    'run086',
    'run084',
    'run082',
    'run081',
]
fe_name = 'stack014'

y_train = pd.read_csv('../input/train.csv')['target']
oofs = [load_oof_from_run_id(ri, to_rank=True) for ri in run_ids]
preds = [load_pred_from_run_id(ri, to_rank=True) for ri in run_ids]

for oof in oofs:
    print(average_precision_score(y_train, oof))

X_train = pd.DataFrame(np.stack(oofs).T)
X_test = pd.DataFrame(np.stack(preds).T)
X_train.columns = run_ids
X_test.columns = run_ids

Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')
