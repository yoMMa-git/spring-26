from typing import Any

import numpy as np

class Node:
    # __slots__ запрещает динамическое добавление атрибутов и убирает __dict__
    # у каждого экземпляра, что экономит память.
    __slots__ = [
        'feature_idx',
        'threshold',
        'children',
        'class_counts',
        'n_samples'
    ]

    def __init__(self, class_counts: np.ndarray, n_samples: float):
        self.feature_idx: int | None = None
        # float-порог для непрерывных разбиений; None — категориальный узел
        self.threshold: float | None = None
        # {значение: Node} для категориальных, {0: левый, 1: правый} для непрерывных
        self.children: dict[float | int, Node] = {}
        self.class_counts: np.ndarray = class_counts
        self.n_samples: float = n_samples

class ID3Tree:
    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Node | None = None
        self.is_categorical: list[bool] | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes: int | None = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, is_categorical: list[bool]):
        self.is_categorical = list(is_categorical)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        weights = np.ones(len(y), dtype=float)
        self.root = self._build(X, y, weights, depth=0)

    def predict(self, X: np.ndarray):
        return np.array([self._predict_one(x) for x in X])

    def prune(self, X_val: np.ndarray, y_val: np.ndarray):
        self._prune_node(self.root, X_val, y_val)

    def depth(self):
        return self._depth(self.root)

    def count_nodes(self):
        return self._count(self.root)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _class_counts(self, y: np.ndarray, weights: np.ndarray):
        """Вернуть взвешенный массив подсчёта классов формы (n_classes,)."""
        return np.array([weights[y == c].sum() for c in self.classes_])

    def _gini(self, counts: np.ndarray):
        """Вычислить индекс Джини по взвешенному массиву подсчёта классов"""
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - float(np.dot(p, p))

    def _weighted_gini(self, splits: list[list]):
        """Вычислить взвешенное среднее Джини по нескольким дочерним разбиениям"""
        total = sum(w for _, w in splits)
        if total == 0: return 0.0
        return sum((w / total) * self._gini(c) for c, w in splits if w > 0)

    def _node_distribution(self, node: Node):
        """Вернуть вектор вероятностей классов, хранящийся в узле"""
        total = node.class_counts.sum()
        if total == 0: return np.ones(self.n_classes) / self.n_classes
        return node.class_counts / total

    def _majority_label(self, node: Node):
        """Вернуть метку класса большинства для данного узла"""
        return self.classes_[int(np.argmax(node.class_counts))]

    def _attach_child(
        self,
        node: Node,
        key: float | int,
        mask: np.ndarray,
        frac: float,
        weights: np.ndarray,
        miss: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ):
        """Сформировать веса дочернего узла и рекурсивно построить его"""
        child_w = np.zeros_like(weights)
        child_w[mask] = weights[mask]
        if miss.any():
            child_w[miss] = weights[miss] * frac
        active = child_w > 0
        if active.sum() > 0:
            node.children[key] = self._build(X[active], y[active], child_w[active], depth + 1)

    def _route_to_child(self, node: Node, key: float | int, x: np.ndarray):
        """Перейти в дочерний узел по ключу или вернуть распределение текущего узла"""
        child = node.children.get(key)
        return self._proba(x, child) if child is not None else self._node_distribution(node)

    def _add_missing_to_splits(
        self, splits: list[list], y_miss: np.ndarray, w_miss: np.ndarray
    ):
        """Добавить пропущенные образцы к splits пропорционально начальным весам ветвей"""
        branch_weights = np.array([s[1] for s in splits])
        total_w = branch_weights.sum()
        for i, s in enumerate(splits):
            frac = branch_weights[i] / total_w if total_w > 0 else 1.0 / len(splits)
            s[0] = s[0] + self._class_counts(y_miss, w_miss * frac)
            s[1] += (w_miss * frac).sum()

    # ------------------------------------------------------------------
    # Поиск наилучшего разбиения
    # ------------------------------------------------------------------

    def _eval_categorical_split(
        self,
        col: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        not_miss: np.ndarray,
        miss: np.ndarray,
    ):
        """Оценить разбиение категориального признака и вернуть его взвешенный Джини"""
        unique_vals = np.unique(col[not_miss])
        if len(unique_vals) <= 1:
            return float('inf'), None

        # Взвешенный подсчёт классов для каждого значения категории
        splits = [
            [
                self._class_counts(y[(col == v) & not_miss], weights[(col == v) & not_miss]),
                weights[(col == v) & not_miss].sum(),
            ]
            for v in unique_vals
        ]

        if miss.any():
            self._add_missing_to_splits(splits, y[miss], weights[miss])

        return self._weighted_gini(splits), None

    def _eval_continuous_split(
        self,
        col: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        not_miss: np.ndarray,
        miss: np.ndarray,
    ):
        """Найти наилучший бинарный порог для непрерывного признака"""
        vals = np.sort(np.unique(col[not_miss]))
        if len(vals) <= 1:
            return float('inf'), None

        best_gini = float('inf')
        best_thresh: float | None = None

        for thresh in (vals[:-1] + vals[1:]) / 2.0:
            left = (col <= thresh) & not_miss
            right = (col > thresh) & not_miss
            lw = weights[left].sum()
            rw = weights[right].sum()
            if lw == 0 or rw == 0:
                continue

            splits = [
                [self._class_counts(y[left], weights[left]), lw],
                [self._class_counts(y[right], weights[right]), rw],
            ]

            if miss.any():
                self._add_missing_to_splits(splits, y[miss], weights[miss])

            gini = self._weighted_gini(splits)
            if gini < best_gini:
                best_gini = gini
                best_thresh = float(thresh)

        return best_gini, best_thresh

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ):
        """Перебрать все признаки и вернуть тот, что минимизирует взвешенный Джини"""
        best_gini = float('inf')
        best_feat: int | None = None
        best_thresh: float | None = None  # None = категориальное разбиение; float = порог непрерывного

        for feat in range(X.shape[1]):
            col = X[:, feat]
            not_miss = ~np.isnan(col)
            miss = ~not_miss

            if not_miss.sum() < 2:
                continue

            if self.is_categorical[feat]:
                gini, thresh = self._eval_categorical_split(col, y, weights, not_miss, miss)
            else:
                gini, thresh = self._eval_continuous_split(col, y, weights, not_miss, miss)

            if gini < best_gini:
                best_gini = gini
                best_feat = feat
                best_thresh = thresh

        return best_feat, best_thresh

    # ------------------------------------------------------------------
    # Построение дерева
    # ------------------------------------------------------------------

    def _should_stop(self, y: np.ndarray, weights: np.ndarray, depth: int):
        """Вернуть True, если рост дерева в этом узле нужно остановить"""
        return (
            len(np.unique(y)) == 1
            or weights.sum() < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )

    def _build_categorical_children(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        col: np.ndarray,
        not_miss: np.ndarray,
        miss: np.ndarray,
        depth: int,
    ):
        """Добавить по одному дочернему узлу на каждое уникальное значение категории"""
        unique_vals = np.unique(col[not_miss])
        not_miss_w = weights[not_miss].sum()

        for v in unique_vals:
            mask = (col == v) & not_miss
            val_w = weights[mask].sum()
            frac = val_w / not_miss_w if not_miss_w > 0 else 1.0 / len(unique_vals)
            self._attach_child(node, float(v), mask, frac, weights, miss, X, y, depth)

    def _build_continuous_children(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        col: np.ndarray,
        not_miss: np.ndarray,
        miss: np.ndarray,
        thresh: float,
        depth: int,
    ):
        """Добавить левый (ключ=0) и правый (ключ=1) дочерние узлы для бинарного порогового разбиения"""
        node.threshold = thresh

        left_mask = (col <= thresh) & not_miss
        right_mask = (col > thresh) & not_miss
        lw = weights[left_mask].sum()
        rw = weights[right_mask].sum()
        total_w = lw + rw
        fl = lw / total_w if total_w > 0 else 0.5
        fr = 1.0 - fl

        for key, mask, frac in ((0, left_mask, fl), (1, right_mask, fr)):
            self._attach_child(node, key, mask, frac, weights, miss, X, y, depth)

    def _build(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, depth: int):
        """Рекурсивно построить дерево решений"""
        class_counts = self._class_counts(y, weights)
        node = Node(class_counts=class_counts, n_samples=float(weights.sum()))

        if self._should_stop(y, weights, depth):
            return node

        feat, thresh = self._best_split(X, y, weights)
        if feat is None:
            return node

        node.feature_idx = feat

        col = X[:, feat]
        not_miss = ~np.isnan(col)
        miss = ~not_miss

        if thresh is None:
            # Категориальный признак: по одной ветви на каждое уникальное значение
            self._build_categorical_children(node, X, y, weights, col, not_miss, miss, depth)
        else:
            # Непрерывный признак: бинарное разбиение по найденному порогу
            self._build_continuous_children(node, X, y, weights, col, not_miss, miss, thresh, depth)

        return node

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def _predict_one(self, x: np.ndarray):
        """Предсказать метку класса для одного образца"""
        proba = self._proba(x, self.root)
        return self.classes_[int(np.argmax(proba))]

    def _proba(self, x: np.ndarray, node: Node):
        """Рекурсивно вычислить вектор вероятностей классов для образца x"""
        if not node.children:
            return self._node_distribution(node)

        val = float(x[node.feature_idx])

        if np.isnan(val):
            # Пропущенное значение: взвешенное среднее по всем дочерним узлам
            total_w = node.n_samples
            result = np.zeros(self.n_classes)
            for child in node.children.values():
                w = child.n_samples / total_w if total_w > 0 else 1.0 / len(node.children)
                result += w * self._proba(x, child)
            return result

        if node.threshold is None:
            # Категориальный узел: маршрутизация по значению признака
            return self._route_to_child(node, val, x)

        # Непрерывный узел: направить влево (<=) или вправо (>)
        return self._route_to_child(node, 0 if val <= node.threshold else 1, x)

    # ------------------------------------------------------------------
    # Сокращение дерева (Reduced Error Pruning)
    # ------------------------------------------------------------------

    def _prune_node(self, node: Node, X_val: np.ndarray, y_val: np.ndarray):
        """Сокращение дерева снизу вверх методом Reduced Error Pruning"""
        if not node.children or len(y_val) == 0:
            return

        col = X_val[:, node.feature_idx]
        not_miss = ~np.isnan(col)

        # Рекурсия в дочерние узлы с соответствующими подмножествами валидации
        if node.threshold is None:
            for v, child in list(node.children.items()):
                mask = (col == v) & not_miss
                if mask.any():
                    self._prune_node(child, X_val[mask], y_val[mask])
        else:
            left_mask = (col <= node.threshold) & not_miss
            right_mask = (col > node.threshold) & not_miss
            if 0 in node.children and left_mask.any():
                self._prune_node(node.children[0], X_val[left_mask], y_val[left_mask])
            if 1 in node.children and right_mask.any():
                self._prune_node(node.children[1], X_val[right_mask], y_val[right_mask])

        # Сравнить ошибку поддерева и ошибку листа на текущем подмножестве валидации
        subtree_errors = np.sum(
            np.array([self._predict_one(x) for x in X_val]) != y_val
        )
        leaf_errors = np.sum(self._majority_label(node) != y_val)

        # Свернуть в лист (очистить children), если это не ухудшает точность
        if leaf_errors <= subtree_errors:
            node.children = {}

    # ------------------------------------------------------------------
    # Статистика дерева
    # ------------------------------------------------------------------

    def _depth(self, node: Node):
        """Вернуть максимальную глубину поддерева с корнем в node"""
        if not node.children: return 0
        return 1 + max(self._depth(c) for c in node.children.values())

    def _count(self, node: Node):
        """Вернуть общее количество узлов в поддереве с корнем в node"""
        if not node.children: return 1
        return 1 + sum(self._count(c) for c in node.children.values())
