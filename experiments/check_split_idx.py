import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# load
train = pd.read_csv('../input/train.csv')

# concat oof
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
train['fold_id'] = np.nan
for i, (tr_idx, val_idx) in enumerate(cv.split(train, train['target'])):
    train.loc[val_idx, 'fold_id'] = i

train['fold_id'].to_csv('../input/fold_id.csv', index=False)
