import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from ayniy.utils import Data


def f(x):
    pred = 0
    for i, d in enumerate(data):
        if i < len(x):
            pred += d[0] * x[i]
        else:
            pred += d[0] * (1 - sum(x))
    score = -1 * average_precision_score(y_train, pred)
    Data.dump(pred, f'../output/pred/{run_name}-train.pkl')
    return score


def make_predictions(data: list, weights: list):
    pred = 0
    for i, d in enumerate(data):
        if i < len(weights):
            pred += d[1] * weights[i]
        else:
            pred += d[1] * (1 - sum(weights))
    Data.dump(pred, f'../output/pred/{run_name}-test.pkl')
    return pred


def make_submission(pred, run_name: str):
    sub = pd.read_csv('../input/atmaCup5__sample_submission.csv')
    sub['target'] = pred
    sub.to_csv(f'../output/submissions/submission_{run_name}.csv', index=False)


run_name = 'weight018'

train = pd.read_csv('../input/train.csv')
y_train = train['target']
data = []

# kmat
dirnames = [
    'result_0603_01_upf0',
    'result_0605_25',
    'result_0605_26',
    'result_0605_27',
]

for dirname in dirnames:
    kmat_preds = []
    kmat_oofs = []
    for i in range(5):
        kmat_preds.append(pd.read_pickle(f'../input/kmat/{dirname}/test_fold{i}.pickle'))
        kmat_oofs.append(pd.read_pickle(f'../input/kmat/{dirname}/val_fold{i}.pickle'))
    kmat_pred = np.mean(kmat_preds, axis=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    train['oof'] = np.nan
    for i, (tr_idx, val_idx) in enumerate(cv.split(train, train['target'])):
        train.loc[val_idx, 'oof'] = kmat_oofs[i]
    data.append((train['oof'].copy().values, kmat_pred))

# print cv score
for d in data:
    print(average_precision_score(y_train, d[0]))

init_state = [round(1 / len(data), 3) for _ in range(len(data) - 1)]
result = minimize(f, init_state, method='Nelder-Mead')
print('optimized CV: ', -1 * result['fun'])
print('w: ', result['x'])
make_submission(make_predictions(data, result['x']), run_name)
