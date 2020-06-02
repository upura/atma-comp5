import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# load
train = pd.read_csv('../input/train.csv')

oofs = []
preds = []

dirname = 'result_0601_02_re'
for i in range(1, 6):
    oofs.append(pd.read_pickle(f'../input/{dirname}/val_fold{i}.pickle'))
    preds.append(pd.read_pickle(f'../input/{dirname}/test_fold{i}.pickle'))

# concat oof
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
train['oof'] = np.nan
for i, (tr_idx, val_idx) in enumerate(cv.split(train, train['target'])):
    train.loc[val_idx, 'oof'] = oofs[i]
    print(average_precision_score(
        train.loc[val_idx, 'target'],
        train.loc[val_idx, 'oof']))

# average test
pred = np.mean(preds, axis=0)

# save
# np.save('oof_result-0601-02', train['oof'].values)
# np.save('pred_result-0601-02', pred)
