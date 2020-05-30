import pandas as pd
from sklearn.metrics import average_precision_score

from ayniy.utils import Data


train = pd.read_csv('../input/train.csv')

oof_005 = Data.load(f'../output/pred/run005-train.pkl')
oof_006 = Data.load(f'../output/pred/run006-train.pkl')
oof_007 = Data.load(f'../output/pred/run007-train.pkl')

print(average_precision_score(train['target'], oof_005))
print(average_precision_score(train['target'], oof_006))
print(average_precision_score(train['target'], oof_007))
print(average_precision_score(train['target'], (oof_005 + oof_007) / 2))
