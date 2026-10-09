# Лабораторная работа №2: Продвинутая инженерия признаков и отбор

## Цель
Освоить продвинутую обработку пропусков, детекцию выбросов, генерацию и отбор признаков
и собрать всё это в единый конвейер машинного обучения.

## Вариант и данные
- **Вариант №1** — **Titanic** (`seaborn`), бинарная классификация (таргет `survived`).
- Ноутбук: `src/notebooks/laboratory_2.ipynb` (выполнен: все таблицы, графики и выводы внутри).
- Данные: `src/data/raw/titanic.csv` — тот же файл, что и в ЛР1, под DVC (повторно из сети не выгружается).
  Пропуски **уже есть**, поэтому условие задания «если пропусков нет…» не выполняется: 10% **MCAR**
  в `age` и `fare` (внесены в ЛР1) плюс родные пропуски Titanic (77% в `deck`). Выбросы тоже есть —
  реальные (`fare`: медиана 14.45, max 512.33), добавлять их не требуется.

## Что реализовано (по шагам задания)

| Шаг | Реализация |
|---|---|
| Импутация | `SimpleImputer(median)`, `KNNImputer`, `IterativeImputer` внутри `Pipeline`, `StratifiedKFold(5)`, метрика ROC-AUC + гистограммы «до/после» |
| Детекция выбросов | `IsolationForest` + визуализация через PCA (2D) + кастомный `OutlierDropper` — sampler для `imblearn.pipeline.Pipeline`, удаляет выбросы **только из train** |
| Генерация признаков | `TitanicFeatureAdder` (`family_size`, `is_alone`, `fare_per_person`, `sex_class`, `deck=Unknown`) и `PolynomialFeatures(degree=2)` (35 → 71 признак) |
| Отбор признаков | Mutual Information (`SelectKBest`) — фильтр, Lasso (`SelectFromModel(LassoCV)`) — встроенный, плюс их union |
| Финальная модель | сквозной пайплайн + `GridSearchCV` (импутер × `contamination` × метод отбора) + оценка на отложенных 25% |

## Результаты (одинаковые 5 фолдов CV, ROC-AUC)

- **Импутация**: KNN(k=5) **0.8625** > Iterative 0.8544 ≈ Simple(медиана) 0.8543.
- **Выбросы**: без удаления 0.8625 > IsolationForest 5% 0.8612 > IsolationForest 10% 0.8562.
- **Признаки**: все без полиномов 0.8657 > все + полиномы 0.8625 > Lasso 0.8602 > MI ∪ Lasso 0.8601 > MI(15) 0.8480.

Финальная конфигурация (`GridSearchCV`, 36 конфигураций): `KNNImputer(k=10)` +
`OutlierDropper(contamination=0.02)` + Lasso-отбор + `RandomForest(max_depth=8)`,
CV ROC-AUC **0.8658**.

| Метрика (отложенные 223 объекта) | базовый пайплайн | финальный пайплайн | прирост |
|---|---|---|---|
| ROC-AUC | **0.8404** | 0.8266 | −0.0139 |
| Accuracy (порог 0.5) | **0.7982** | 0.7399 | −0.0583 |
| F1 (класс «выжил») | **0.7170** | 0.6420 | −0.0750 |

**Честный итог:** выигрыш на кросс-валидации не подтвердился на отложенной выборке — на 223
объектах различия лежат внутри статистического шума, и усложнение пайплайна не гарантирует
улучшения. Разбор причин (малый тест, выбор из 36 конфигураций по CV, неоптимальный порог 0.5)
и рекомендация, какие стадии стоит оставлять в проде, — в разделе 9.4 ноутбука.

## Как запустить
```bash
# 1) зависимости
.venv\Scripts\activate
pip install -r requirements.txt

# 2) выполнить ноутбук — интерактивно или целиком headless
jupyter lab src/notebooks/laboratory_2.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace src/notebooks/laboratory_2.ipynb

# 3) данные уже под контролем DVC (src/data/raw/titanic.csv из ЛР1), при необходимости:
dvc pull
```
Интерактивно: `jupyter lab src/notebooks/laboratory_2.ipynb`.

Артефакты выполнения не коммитятся (создаются ноутбуком заново):
`src/reports/figures/*.png`, `src/reports/*.csv`, `src/models/lab2_titanic_pipeline.joblib`.
