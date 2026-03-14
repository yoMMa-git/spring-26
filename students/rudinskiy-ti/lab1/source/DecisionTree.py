import numpy as np

def gini(y):
    p = (y == 1).mean()
    return 4 * p * (1 - p)

class Node:
    def __init__(self, predicted_class=None, is_leaf = None, threshold = None, feature = None):
        self.predicted_class = predicted_class
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.feature = feature
        self.left = None
        self.right = None
        self.class_probs = None
        
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, tol=0.01, to_prune = True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tol = tol
        self.to_prune = to_prune

    def fit(self, X, y, X_val = None, y_val = None):
        self.classes = np.unique(y)
        self.tree_ = self._grow_tree(X, y)
        if self.to_prune and X_val is not None and y_val is not None:
            self._prune_node(self.tree_, X_val, y_val)

    def predict(self, X):
        arr = np.array(X)
        return [self._predict(inputs) for inputs in arr]

    def _major(self, y):
        vals, cnts = np.unique(y, return_counts=True)
        major_y = vals[cnts.argmax()]
        return major_y

    def _gain(self, U, Uy, feature):
        mask = ~np.isnan(U[:, feature])
        U_m = U[mask]
        Uy_m = Uy[mask]

        if len(Uy_m) == 0:
            return 0, None

        Phi = gini(Uy_m)
        U_len = U_m.shape[0]
        Q_max = -1e-10
        threshold = 0
        feature_vals =  np.unique(U_m[:, feature])

        for i in feature_vals:
            left_mask =  U_m[:, feature] <= i
            right_mask = ~left_mask
            left_y = Uy_m[left_mask]
            right_y = Uy_m[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            Q_curr = Phi - (len(left_y) * gini(left_y) + len(right_y) * gini(right_y)) / U_len
            if Q_curr > Q_max:
              Q_max = Q_curr
              threshold = i

        return threshold, Q_max

    def _count_err(self, node, Xq, yq):
        if len(yq) == 0:
            return 0

        if node.is_leaf:
            return len(yq) - np.sum(node.predicted_class == yq)

        err = 0
        for i in range(len(Xq)):
            if Xq[i][node.feature] <= node.threshold:
                err += self._count_err(node.left, [Xq[i]], [yq[i]])
            else:
                err += self._count_err(node.right, [Xq[i]], [yq[i]])
        return err

    def _prune_node(self, node, Xq, yq):

        if len(Xq) == 0:
            node.is_leaf = True
            node.left = None
            node.right = None
            if len(yq) > 0:
                node.class_probs = {i: np.sum(yq == i) / len(yq) for i in self.classes}
            return

        if node.is_leaf:
          return

        if node.left:
            left_mask = Xq[:, node.feature] <= node.threshold
            self._prune_node(node.left, Xq[left_mask], yq[left_mask])
        if node.right:
            right_mask = Xq[:, node.feature] > node.threshold
            self._prune_node(node.right, Xq[right_mask], yq[right_mask])

        err1 = self._count_err(node, Xq, yq)
        err2 = len(yq) - np.sum(node.predicted_class == yq)
        err3 = self._count_err(node.left, Xq, yq) if node.left else 1e6
        err4 = self._count_err(node.right, Xq, yq) if node.right else 1e6

        errs = np.array([err1, err2, err3, err4])
        z = np.argmin(errs)
        if z == 1:
            node.is_leaf = True
            node.left = None
            node.right = None
            if len(yq) > 0:
                node.class_probs = {i: np.sum(yq == i) / len(yq) for i in self.classes}
        elif z == 2:
            if node.left.is_leaf:
                node.is_leaf = True
                node.left = None
                node.right = None
                node.class_probs = node.left.class_probs if node.left else node.class_probs
            else:
                old_node = node.left
                node.feature = old_node.feature
                node.threshold = old_node.threshold
                node.left = old_node.left
                node.right = old_node.right
                node.is_leaf = old_node.is_leaf
                node.class_probs = old_node.class_probs
        elif z == 3:
            if node.right.is_leaf:
                node.is_leaf = True
                node.left = None
                node.right = None
                node.class_probs = node.right.class_probs  if node.right else node.class_probs
            else:
                old_node = node.right
                node.feature = old_node.feature
                node.threshold = old_node.threshold
                node.left = old_node.left
                node.right = old_node.right
                node.is_leaf = old_node.is_leaf
                node.class_probs = old_node.class_probs

    def _grow_tree(self, U, y, depth=0):
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split):
            node = Node(predicted_class=self._major(y), is_leaf = True)
            node.class_probs = {}
            for i in self.classes:
                node.class_probs[i] = np.sum(y == i) / len(y)
            return node

        Q_max = -1e-10
        thr_max = None
        f_max = None
        U_len, U_width = U.shape

        for feature in range(U_width):
            thr, Q_f = self._gain(U, y, feature)
            if Q_f is not None and Q_f > Q_max:
              Q_max = Q_f
              thr_max = thr
              f_max = feature
        mask = ~np.isnan(U[:, f_max])

        if Q_max < self.tol:
            major_y = self._major(y)
            node = Node(predicted_class=major_y, is_leaf=True)
            node.class_probs = {}
            for i in self.classes:
                node.class_probs[i] = np.sum(y == i) / len(y)
            return node

        major_y = self._major(y)

        node = Node(predicted_class=major_y)
        node.feature = f_max
        node.threshold = thr_max
        node.is_leaf = False
        left_mask = (U[:, f_max] <= thr_max) & mask
        right_mask = (~left_mask) & mask
        node.q_left = np.sum(left_mask) / U_len
        node.q_right = np.sum(right_mask) / U_len
        node.left = self._grow_tree(U[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(U[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_proba(self, inputs, node = None):
        if node is None:
            node = self.tree_

        if node.is_leaf:
            return node.class_probs

        if np.isnan(inputs[node.feature]):
            probs = {i: 0.0 for i in self.classes}

            if node.left:
                left_p = self._predict_proba(inputs, node.left)
                for i in self.classes:
                    probs[i] += node.q_left * left_p[i]

            if node.right:
                right_p = self._predict_proba(inputs, node.right)
                for i in self.classes:
                    probs[i] += node.q_right * right_p[i]

            return probs

        if inputs[node.feature] <= node.threshold:
            return self._predict_proba(inputs, node.left)
        else:
            return self._predict_proba(inputs, node.right)

    def _predict(self, inputs):
          arr = self._predict_proba(inputs)
          return max(arr, key=arr.get)
    
