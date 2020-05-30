import pandas as pd
from sklearn.metrics import average_precision_score

from ayniy.utils import Data


train = pd.read_csv('../input/train.csv')

oof_008 = Data.load(f'../output/pred/run008-train.pkl')
oof_009 = Data.load(f'../output/pred/run009-train.pkl')

print(average_precision_score(train['target'], oof_008))
print(average_precision_score(train['target'], oof_009))
print(average_precision_score(train['target'], (oof_008 + oof_009) / 2))
