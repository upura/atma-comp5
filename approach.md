# Approach

## Feature Engineering

- label_encoding
- frequency encoding
- count encoding
- count encoding interact
- matrix factorization
- aggregation
- numeric interact
- tsfresh
- target encoding

Then, 200 features are selected by lgbm importance.

## Validation Strategy

GroupKFold does't work because a fold has few positive data. 

- 5 StratifiedKFold
- 6 GroupKFold

## Model

### GBDT

- LightGBM
- CatBoost
- XGBoost

Logloss works better than PR-AUC as a eval metric

### NN

- CNN
    - [Kaggle "PLAsTiCC Astronomical Classification" 3rd place](https://www.kaggle.com/yuval6967/3rd-place-cnn)
- RNN
    - [Kaggle "PLAsTiCC Astronomical Classification" 2nd place](https://www.kaggle.com/zerrxy/plasticc-rnn)

## Ensemble

- Weighted averaging
- Stacking
