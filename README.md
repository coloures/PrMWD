# Лабораторная работа №1: Введение и воспроизводимость исследований

## Цель
Развернуть воспроизводимый Data Science-проект и обучить модель бинарной
классификации выживания пассажиров «Титаника». Работа демонстрирует полный
жизненный цикл исследования: подготовка грязных данных, кастомные трансформеры,
конвейер предобработки + модели и экспорт обученной модели.

## Вариант и данные
- **Вариант №1** — датасет **Titanic** (библиотека `seaborn`).
- **Задача** — бинарная классификация (таргет: `survived`).
- Для имитации «грязных» данных в 10% случайных значений колонок `age` и `fare`
  искусственно вносятся пропуски; результат сохраняется в `src/data/raw/titanic.csv`.

## Стек технологий
- `pandas` — работа с данными
- `seaborn` — загрузка датасета
- `scikit-learn` — предобработка (SimpleImputer, StandardScaler, OneHotEncoder),
  кастомный трансформер, Pipeline, ColumnTransformer, RandomForestClassifier
- `joblib` — сериализация обученной модели
- `jupyterlab` — интерактивная среда для ноутбука

## Структура проекта
```
Pract_exercise/
├── models/                      # обученные модели (*.joblib)
├── src/
│   ├── data/
│   │   └── raw/                 # исходные данные (titanic.csv), версии — в DVC
│   ├── features/                # зарезервировано под генерацию признаков
│   ├── models/                  # зарезервировано под код моделей
│   ├── notebooks/
│   │   └── laboratory_1.ipynb   # основная рабочая тетрадь (вся логика — здесь)
│   ├── reports/                 # отчёты и скриншоты
│   └── visualization/           # зарезервировано под визуализацию
├── .gitignore
└── requirements.txt
```

## Как запустить
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# выполнить ноутбук целиком (загрузка данных, пропуски, трансформер,
# пайплайн, обучение, метрики, экспорт модели)
jupyter lab src/notebooks/laboratory_1.ipynb
```
После выполнения обученный пайплайн сохраняется в `models/titanic_model.joblib`,
выводятся метрики **Accuracy** и **ROC-AUC**.

## Метрики (тестовая выборка, 80/20, random_state=42)
- **Accuracy: 0.8212**
- **ROC-AUC: 0.8555**

## DVC
Датасет `src/data/raw/titanic.csv` добавлен под контроль версий DVC:
```bash
dvc add src/data/raw/titanic.csv
git add src/data/raw/titanic.csv.dvc src/data/raw/.gitignore
git commit -m "lab1: track titanic.csv with DVC"
dvc status   # Data and pipelines are up to date
```