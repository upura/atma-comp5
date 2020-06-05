import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from ayniy.utils import Data


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if run_id in ('run015'):
        oof = oof.reshape(-1, )
        pred = pred.reshape(-1, )
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


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


run_ids = [
    'weight016',
    'weight019',
    'weight020',
]
run_name = 'weight021'

y_train = pd.read_csv('../input/train.csv')['target']
data = [load_from_run_id(ri, to_rank=True) for ri in run_ids]

for d in data:
    print(average_precision_score(y_train, d[0]))

init_state = [round(1 / len(data), 3) for _ in range(len(data) - 1)]
result = minimize(f, init_state, method='Nelder-Mead')
print('optimized CV: ', -1 * result['fun'])
print('w: ', result['x'])
make_submission(make_predictions(data, result['x']), run_name)
