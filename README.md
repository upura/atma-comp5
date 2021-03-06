# atmaCup #5

- [atmaCup #5](https://atma.connpass.com/event/175139/)のソースコード。最終順位はpublic 16位->private 27位。
- 取り組みについては、[ブログ](https://upura.hatenablog.com/entry/2020/06/06/193944)や[approach.md](approach.md)をご覧ください。
- 自作コンペ用ライブラリ「Ayniy」を利用しています。

[**Documentation**](https://upura.github.io/ayniy-docs/) | [**GitHub**](https://github.com/upura/ayniy) | [**Slide (Japanese)**](https://speakerdeck.com/upura/ayniy-with-mlflow)

```python
# Import packages
import yaml
from sklearn.model_selection import StratifiedKFold
from ayniy.preprocessing.runner import Tabular
from ayniy.model.runner import Runner

# Load configs
f = open('configs/fe000.yml', 'r+')
fe_configs = yaml.load(f)
g = open('configs/run000.yml', 'r+')
run_configs = yaml.load(g)

# Difine CV strategy as you like
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# Feature engineering
tabular = Tabular(fe_configs, cv)
tabular.create()

# Modeling
runner = Runner(run_configs, cv)
runner.run_train_cv()
runner.run_predict_cv()
runner.submission()
```

## Environment

```bash
docker-compose build
docker-compose up
```

## MLflow

```bash
cd experiments
mlflow ui
```

## Test

```bash
docker-compose build
docker-compose up -d
docker exec -it ayniy-test bash
```
``` 
pytest tests/ --cov=. --cov-report=html
```

## Docs

```bash
docker-compose build
docker-compose up -d
docker exec -it ayniy-test bash
cd docs
make html
```
```bash
cd docs/build/html
git a .
git c "update"
git push origin master
```
https://github.com/upura/ayniy-docs
