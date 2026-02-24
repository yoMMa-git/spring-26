import numpy as np
import pandas as pd


def _gini(y: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    p1 = np.sum(y == 1) / n
    return 2 * p1 * (1 - p1)


def _majority(y: np.ndarray) -> int:
    if len(y) == 0:
        return 0
    return np.argmax(np.bincount(y.astype(int)))


def _class_distribution(y: np.ndarray) -> dict:
    n = len(y)
    if n == 0:
        return {0: 0.5, 1: 0.5}
    cnt = np.bincount(y.astype(int), minlength=2)
    return {0: cnt[0] / n, 1: cnt[1] / n}


class Node:
    def __init__(self, is_leaf: bool = False):
        self.is_leaf = is_leaf
        self.class_probs: dict[int, float] = None  # {0: p0, 1: p1}
        self.label: int = None  # argmax class
        self.n_samples: int = 0
        self.feature_name: str = None
        self.feature_idx: int = None
        self.split_type: str = None  # 'numeric' | 'categorical'
        self.threshold: float = None  # для numeric
        self.values: dict[str, str] = None  # для categorical: множество значений ветви
        self.children: dict[str, Node] = None
        self.q_vk: dict[str, float] = None


class DecisionTree:
    def __init__(
        self,
        min_gain: float = 1e-6,
        min_samples_leaf: int = 1,
        max_depth: int = None,
    ):
        self.min_gain = min_gain
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.root = None
        self.feature_names_ = None
        self.feature_types_ = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_types: dict = None,
    ):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)
        self.feature_types_ = feature_types or {c: "numeric" for c in self.feature_names_}
        y = np.asarray(y).ravel()
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int) -> Node:
        n = len(y)
        if n == 0:
            node = Node(is_leaf=True)
            node.class_probs = {0: 0.5, 1: 0.5}
            node.label = 0
            node.n_samples = 0
            return node

        gini_u = _gini(y)
        if gini_u < 1e-10 or (self.max_depth is not None and depth >= self.max_depth):
            node = Node(is_leaf=True)
            node.class_probs = _class_distribution(y)
            node.label = _majority(y)
            node.n_samples = n
            return node

        best_gain = -1.0
        best_feature = None
        best_split = None

        for col in self.feature_names_:
            idx = self.feature_names_.index(col)
            ftype = self.feature_types_.get(col, "numeric")
            known = X[col].notna()
            U_known = X.loc[known]
            y_known = y[known.values]
            n_known = len(y_known)
            if n_known < self.min_samples_leaf * 2:
                continue
            gini_known = _gini(y_known)
            if ftype == "numeric":
                gain, split_info = self._best_numeric_split(U_known, y_known, col, X, y)
            else:
                gain, split_info = self._best_categorical_split(U_known, y_known, col, X, y)
            if split_info is None:
                continue
            if gain > best_gain and gain >= self.min_gain:
                best_gain = gain
                best_feature = col
                best_split = split_info

        if best_feature is None or best_gain < self.min_gain:
            node = Node(is_leaf=True)
            node.class_probs = _class_distribution(y)
            node.label = _majority(y)
            node.n_samples = n
            return node

        col = best_feature
        idx = self.feature_names_.index(col)
        ftype = self.feature_types_.get(col, "numeric")
        split_type, split_val, partitions = best_split  # partitions: list of (branch_key, indices)

        node = Node(is_leaf=False)
        node.feature_name = col
        node.feature_idx = idx
        node.split_type = ftype
        if ftype == "numeric":
            node.threshold = split_val
        else:
            node.values = split_val  # dict or set of (value -> branch_key)
        node.n_samples = n

        # Оценки вероятностей ветвей q_vk по объектам с известным значением
        known = X[col].notna()
        n_known = known.sum()
        node.q_vk = {}
        node.children = {}

        for branch_key, inds in partitions:
            if len(inds) < self.min_samples_leaf:
                continue
            X_sub = X.iloc[inds]
            y_sub = y[inds]
            node.q_vk[branch_key] = len(inds) / max(n_known, 1)
            node.children[branch_key] = self._grow_tree(X_sub, y_sub, depth + 1)

        # Нормализовать q_vk
        total_q = sum(node.q_vk.values())
        if total_q > 0:
            for k in node.q_vk:
                node.q_vk[k] /= total_q
        return node

    def _get_numeric_branches(self, X: pd.DataFrame, col: str, threshold: float):
        """Разбиение по порогу: left = <= threshold, right = > threshold"""
        left = X[col] <= threshold
        right = ~left
        return [("left", np.where(left.values)[0]), ("right", np.where(right.values)[0])]

    def _best_numeric_split(self, U: pd.DataFrame, y_u: np.ndarray, col: str, X_full: pd.DataFrame, y_full: np.ndarray):
        """Перебор порогов по уникальным значениям; возврат (best_gain, (split_type, threshold, partitions))"""
        vals = U[col].values
        uniq = np.unique(vals)
        if len(uniq) < 2:
            return -1.0, None
        thresholds = (uniq[:-1] + uniq[1:]) / 2
        best_gain = -1.0
        best_partitions = None
        best_thr = None
        n = len(y_u)
        gini_u = _gini(y_u)
        for th in thresholds:
            left = vals <= th
            right = ~left
            n_l, n_r = left.sum(), right.sum()
            if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                continue
            gini_l = _gini(y_u[left])
            gini_r = _gini(y_u[right])
            gain = gini_u - (n_l / n) * gini_l - (n_r / n) * gini_r
            idx_known = U.index
            il = np.where(U[col].values <= th)[0]
            ir = np.where(U[col].values > th)[0]
            inds_l = X_full.index.get_indexer(idx_known[il])
            inds_r = X_full.index.get_indexer(idx_known[ir])
            inds_l, inds_r = inds_l[inds_l >= 0], inds_r[inds_r >= 0]
            if len(inds_l) < self.min_samples_leaf or len(inds_r) < self.min_samples_leaf:
                continue
            if gain > best_gain:
                best_gain = gain
                best_thr = th
                best_partitions = [("left", inds_l), ("right", inds_r)]
        if best_partitions is None:
            return -1.0, None
        return best_gain, ("numeric", best_thr, best_partitions)

    def _best_categorical_split(self, U: pd.DataFrame, y_u: np.ndarray, col: str, X_full: pd.DataFrame, y_full: np.ndarray):
        """Многозначное разбиение по категориям. Возврат (gain, (split_type, values_set, partitions))"""
        vals = U[col].values
        uniq = np.unique(vals)
        uniq = uniq[~np.isnan(uniq)] if np.issubdtype(uniq.dtype, np.floating) else uniq
        if len(uniq) < 2:
            return -1.0, None
        n = len(y_u)
        gini_u = _gini(y_u)
        parts = []
        for v in uniq:
            mask = (vals == v) if not (isinstance(v, float) and np.isnan(v)) else np.isnan(vals)
            inds = np.where(mask)[0]
            if len(inds) < self.min_samples_leaf:
                continue
            idx_known = U.index
            inds_full = X_full.index.get_indexer(U.index[inds])
            inds_full = inds_full[inds_full >= 0]
            if len(inds_full) < self.min_samples_leaf:
                continue
            parts.append((v, inds_full))
        if len(parts) < 2:
            return -1.0, None
        weighted_gini = 0.0
        for v, inds in parts:
            u_pos = np.where(vals == v)[0]
            y_sub = y_u[u_pos]
            weighted_gini += (len(y_sub) / n) * _gini(y_sub)
        gain = gini_u - weighted_gini
        if gain < self.min_gain:
            return -1.0, None
        return gain, ("categorical", {p[0]: p[0] for p in parts}, parts)

    def _predict_proba_one(self, x: np.ndarray, node: Node) -> dict:
        """Вероятности классов для одного объекта"""
        if node.is_leaf:
            return node.class_probs.copy()
        col = node.feature_name
        idx = node.feature_idx
        val = x[idx] if isinstance(x, np.ndarray) else x[col]
        is_missing = pd.isna(val) if not isinstance(val, (int, np.integer)) else False
        if is_missing or (isinstance(val, float) and np.isnan(val)):
            # P(y|x,v) = sum_k q_vk * P(y|x, S_v(k))
            probs = {0: 0.0, 1: 0.0}
            for k, child in node.children.items():
                w = node.q_vk.get(k, 0.0)
                cp = self._predict_proba_one(x, child)
                probs[0] += w * cp.get(0, 0)
                probs[1] += w * cp.get(1, 0)
            return probs
        if node.split_type == "numeric":
            branch = "left" if val <= node.threshold else "right"
        else:
            branch = val
        if branch not in node.children:
            return {0: 0.5, 1: 0.5}
        return self._predict_proba_one(x, node.children[branch])

    def predict_proba(self, X):
        """Вероятности для каждого класса (0, 1)"""
        if self.root is None:
            raise ValueError("Tree not fitted")
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names_].values if self.feature_names_ else X.values
        X = np.atleast_2d(X)
        res = np.zeros((len(X), 2))
        for i in range(len(X)):
            p = self._predict_proba_one(X[i], self.root)
            res[i, 0] = p.get(0, 0.5)
            res[i, 1] = p.get(1, 0.5)
        return res

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _count_nodes(self, node: Node) -> tuple:
        """(всего узлов, листьев, глубина)."""
        if node.is_leaf:
            return 1, 1, 0
        total, leaves, depth = 1, 0, 0
        for ch in (node.children or {}).values():
            t, l, d = self._count_nodes(ch)
            total += t
            leaves += l
            depth = max(depth, d + 1)
        return total, leaves, depth + 1

    def get_tree_stats(self) -> dict:
        """Число узлов, листьев, глубина"""
        if self.root is None:
            return {"n_nodes": 0, "n_leaves": 0, "depth": 0}
        n_nodes, n_leaves, depth = self._count_nodes(self.root)
        return {"n_nodes": n_nodes, "n_leaves": n_leaves, "depth": depth}

    def prune(self, X_val, y_val):
        """REP по валидационной выборке"""
        if isinstance(X_val, pd.DataFrame):
            X_val = np.asarray(X_val)
        y_val = np.asarray(y_val).ravel()

        def try_prune(node, parent_branch=None, parent=None) -> bool:
            """Возвращает True если узел был заменён на лист."""
            if node.is_leaf:
                return False

            for k in list(node.children.keys()):
                try_prune(node.children[k], k, node)

            old_preds = self.predict(X_val)
            old_acc = np.mean(old_preds == y_val)
            old_children = node.children
            old_feature = node.feature_name
            old_q_vk = node.q_vk
            val_reach = self._samples_reaching_node(X_val, node)
            if val_reach.sum() == 0:
                return False
            y_reach = y_val[val_reach]
            majority_val = _majority(y_reach)
            node.is_leaf = True
            node.class_probs = _class_distribution(y_reach)
            node.label = majority_val
            node.children = None
            node.feature_name = None
            node.q_vk = None
            new_preds = self.predict(X_val)
            new_acc = np.mean(new_preds == y_val)
            if new_acc >= old_acc:
                return True
            node.is_leaf = False
            node.children = old_children
            node.feature_name = old_feature
            node.q_vk = old_q_vk
            node.class_probs = None
            node.label = None
            return False

        self._prune_recurse(self.root, X_val, y_val)

    def _samples_reaching_node(self, X: np.ndarray, target: Node) -> np.ndarray:
        """Булева маска: какие строки X доходят до target при обходе от root"""
        if self.root is target:
            return np.ones(len(X), dtype=bool)
        return self._reach_mask(X, self.root, target)

    def _reach_mask(self, X: np.ndarray, node: Node, target: Node) -> np.ndarray:
        if node.is_leaf:
            return np.zeros(len(X), dtype=bool)
        out = np.zeros(len(X), dtype=bool)
        col = node.feature_idx
        xcol = X[:, col]
        valid = np.ones(len(xcol), dtype=bool)
        if np.issubdtype(xcol.dtype, np.floating):
            valid = ~np.isnan(xcol)
        for branch_key, child in node.children.items():
            if node.split_type == "numeric":
                if branch_key == "left":
                    mask = valid & (xcol <= node.threshold)
                else:
                    mask = valid & (xcol > node.threshold)
            else:
                mask = valid & (xcol == branch_key)
            if child is target:
                out |= mask
            else:
                sub = self._reach_mask(X[mask], child, target)
                out[mask] = sub
        return out

    def _prune_recurse(self, node: Node, X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Пост-прунинг: сначала обрезаются дети, потом попытка обрезать этот узел. Результат: была ли вершина заменена на лист"""
        if node.is_leaf:
            return False
        for k in list(node.children.keys()):
            self._prune_recurse(node.children[k], X_val, y_val)
        acc_before = np.mean(self.predict(X_val) == y_val)
        reach = self._samples_reaching_node(X_val, node)
        if reach.sum() < 2:
            return False
        y_reach = y_val[reach]
        maj = _majority(y_reach)
        saved_children = node.children
        saved_feature = node.feature_name
        saved_q_vk = node.q_vk
        saved_split_type = node.split_type
        saved_threshold = node.threshold
        node.is_leaf = True
        node.class_probs = _class_distribution(y_reach)
        node.label = maj
        node.children = None
        node.feature_name = None
        node.q_vk = None
        acc_after = np.mean(self.predict(X_val) == y_val)
        if acc_after >= acc_before:
            return True
        node.is_leaf = False
        node.children = saved_children
        node.feature_name = saved_feature
        node.q_vk = saved_q_vk
        node.split_type = saved_split_type
        node.threshold = saved_threshold
        node.class_probs = None
        node.label = None
        return False
