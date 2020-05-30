import pandas as pd
from tsfresh import extract_features


spec_tr = pd.read_csv('../input/spec_tr.csv')
spec_te = pd.read_csv('../input/spec_te.csv')

X_tr = pd.melt(spec_tr.T)
ef_tr = extract_features(X_tr, column_id='variable')

X_te = pd.melt(spec_te.T)
ef_te = extract_features(X_te, column_id='variable')

ef_tr.to_csv('../input/ef_tr.csv')
ef_te.to_csv('../input/ef_te.csv')
