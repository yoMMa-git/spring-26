import logging
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model import ID3Tree
from utils import load_data, evaluate

def run_pipeline(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        is_categorical: list[bool],
        save_path: str,
):
    # Обучаем собственную модель
    logging.info("Обучаем кастомное дерево ID3")
    tree = ID3Tree(max_depth=None, min_samples_split=2)
    tree.fit(X_train, y_train, is_categorical)
    logging.info(f"Глубина: {tree.depth()},  Узлов: {tree.count_nodes()}")

    # Предсказания до редукции
    y_pred_train_before = tree.predict(X_train)
    y_pred_before = tree.predict(X_test)

    path = os.path.join(save_path, "id3_train_before.txt")
    evaluate(y_train, y_pred_train_before, "ID3 train (до редукции)", save_path=path)
    path = os.path.join(save_path, "id3_test_before.txt")
    evaluate(y_test, y_pred_before, "ID3 test (до редукции)", save_path=path)

    # Редукция дерева по валидационной выборке
    logging.info("Применяем Reduced Error Pruning")
    tree.prune(X_val, y_val)
    logging.info(f"Глубина после редукции: {tree.depth()},  Узлов: {tree.count_nodes()}")

    # Предсказания после редукции
    y_pred_train_after = tree.predict(X_train)
    y_pred_after = tree.predict(X_test)

    path = os.path.join(save_path, "id3_train_after.txt")
    evaluate(y_train, y_pred_train_after, "ID3 train (после редукции)", save_path=path)
    path = os.path.join(save_path, "id3_test_after.txt")
    evaluate(y_test, y_pred_after, "ID3 test (после редукции)", save_path=path)

    # Эталонная модель sklearn
    logging.info("Обучаем sklearn DecisionTreeClassifier")
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    sk_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
    sk_tree.fit(X_train_imp, y_train)
    y_pred_sklearn = sk_tree.predict(X_test_imp)

    path = os.path.join(save_path, "sklearn_test.txt")
    evaluate(y_test, y_pred_sklearn, "sklearn DecisionTree – Test", save_path=path)

    # Матрицы ошибок
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_preds = [
        ("ID3 до редукции", y_pred_before),
        ("ID3 после редукции", y_pred_after),
        ("sklearn", y_pred_sklearn),
    ]
    for ax, (title, y_pred) in zip(axes, model_preds):
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrices.png"), dpi=300)
    plt.close()


def main():
    SEED = 42
    np.random.seed(SEED)

    # Создаём директорию для результатов
    results_path = os.path.join(os.getcwd(), "students", "chebykin-aa", "lab1", "results")
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path, exist_ok=True)

    # Настраиваем логирование
    logging.basicConfig(
        filename=os.path.join(results_path, "main.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Загрузка данных
    X, y, _, is_categorical = load_data()

    missing_mask = np.isnan(X)
    n_cat = sum(is_categorical)
    n_cont = len(is_categorical) - n_cat
    logging.info(f"Samples: {X.shape[0]}")
    logging.info(f"Features: {X.shape[1]} ({n_cont} числовых + {n_cat} категориальных)")
    logging.info(f"Классы: выжил=0 ({(y == 0).sum()}), умер=1 ({(y == 1).sum()})")
    logging.info(f"Пропуски: {int(missing_mask.sum())} ({missing_mask.mean() * 100:.1f} %)")

    # Разбиваем данные: 64%/16%/20%
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=SEED, stratify=y_tv)

    logging.info(f"Размер тренировочной выборки: {len(y_train)}")
    logging.info(f"Размер валидационной выборки: {len(y_val)}")
    logging.info(f"Размер тестовой выборки: {len(y_test)}")

    run_pipeline(X_train, X_val, X_test, y_train, y_val, y_test, is_categorical,
                 save_path=results_path)


if __name__ == "__main__":
    main()
