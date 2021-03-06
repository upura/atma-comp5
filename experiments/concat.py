import pandas as pd

from ayniy.utils import Data
# from ayniy.preprocessing import standerize


# sub = pd.read_csv('../input/atmaCup5__sample_submission.csv')
# train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/test.csv')
# fitting = pd.read_csv('../input/fitting.csv')

# train = pd.merge(train, fitting, on='spectrum_id', how='inner')
# test = pd.merge(test, fitting, on='spectrum_id', how='inner')

# train.to_csv('../input/train_fitting.csv', index=False)
# test.to_csv('../input/test_fitting.csv', index=False)

add_tr = pd.read_csv('../input/additional_features_train.csv')
add_te = pd.read_csv('../input/additional_features_test.csv')

fe005_tr = Data.load('../input/X_train_fe005.pkl')
fe005_te = Data.load('../input/X_test_fe005.pkl')

# fe001_tr = Data.load('../input/X_train_fe001.pkl')
# fe001_te = Data.load('../input/X_test_fe001.pkl')
# top10_tr = Data.load('../input/X_train_fe004_top10.pkl')
# top10_te = Data.load('../input/X_test_fe004_top10.pkl')

# top10_tr, top10_te = standerize(top10_tr, top10_te, {'encode_col': top10_tr.columns})
# print(top10_tr.head())

train_fitting_ef_add = pd.concat([fe005_tr, add_tr], axis=1)
test_fitting_ef_add = pd.concat([fe005_te, add_te], axis=1)

fe_name = 'fe005_add'
Data.dump(train_fitting_ef_add, f'../input/X_train_{fe_name}.pkl')
Data.dump(test_fitting_ef_add, f'../input/X_test_{fe_name}.pkl')

# train_fitting_ef.to_csv('../input/train_fitting_ef.csv', index=False)
# test_fitting_ef.to_csv('../input/test_fitting_ef.csv', index=False)

# spec_train = []
# for i, filename in enumerate(train['spectrum_filename'].values):
#     spec_df = pd.read_csv(f'../input/spectrum_raw/{filename}', sep='\t', header=None)
#     spec_train.append(spec_df[1])

# spec_tr = pd.concat(spec_train, axis=1)
# spec_tr = spec_tr.T
# spec_tr.columns = [f'spec_{i}' for i in range(spec_tr.shape[1])]
# spec_tr = spec_tr.reset_index(drop=True)

# train = pd.concat([train, spec_tr], axis=1)

# spec_test = []
# for i, filename in enumerate(test['spectrum_filename'].values):
#     spec_df = pd.read_csv(f'../input/spectrum_raw/{filename}', sep='\t', header=None)
#     if len(spec_df[1]) == 512:
#         spec_test.append(spec_df[1])
#     else:
#         spec_test.append(pd.Series([0] + list(spec_df[1].values)))

# spec_te = pd.concat(spec_test, axis=1)
# spec_te = spec_te.T
# spec_te.columns = [f'spec_{i}' for i in range(spec_te.shape[1])]
# spec_te = spec_te.reset_index(drop=True)

# test = pd.concat([test, spec_te], axis=1)

# train.to_csv('../input/train__fitting_spec.csv', index=False)
# test.to_csv('../input/test_fitting_spec.csv', index=False)

# spec_tr.to_csv('../input/spec_tr.csv', index=False)
# spec_te.to_csv('../input/spec_te.csv', index=False)
