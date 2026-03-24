from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, 
                 value=None, class_probs=None, split_type='numeric', children=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.children = children or {}
        self.split_type = split_type
        self.value = value
        self.class_probs = class_probs or {}
        self.n_samples = 0


class DecisionTree:
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.classes_ = None
        self.feature_types_ = None  # {feature_idx: 'numeric'|'categorical'}
    
    def gini(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        n = len(y)
        g = 1.0
        for c in counts.values():
            p = c / n
            g -= p * p
        return g
    
    def _compute_class_probs(self, y):
        if len(y) == 0:
            return {}
        counts = Counter(y)
        n = len(y)
        return {cls: count / n for cls, count in counts.items()}
    
    def _best_numeric_split(self, X, y, feature):
        valid = [(X[i][feature], y[i]) for i in range(len(X)) if X[i][feature] is not None]
        
        if len(valid) < 2:
            return float('inf'), None
        
        values = sorted(set(v for v, _ in valid))
        if len(values) < 2:
            return float('inf'), None
        
        best_gini = float('inf')
        best_threshold = None
        n_total = len(y)
        
        for i in range(len(values) - 1):
            t = (values[i] + values[i + 1]) / 2
            
            left = [label for v, label in valid if v <= t]
            right = [label for v, label in valid if v > t]
            
            if not left or not right:
                continue
            
            g = (len(left)/n_total) * self.gini(left) + (len(right)/n_total) * self.gini(right)
            
            if g < best_gini:
                best_gini = g
                best_threshold = t
        
        return best_gini, best_threshold

    def _best_categorical_split(self, X, y, feature):
        valid = [(X[i][feature], y[i]) for i in range(len(X)) if X[i][feature] is not None]
        
        if len(valid) < 2:
            return float('inf'), None
        
        categories = list(set(v for v, _ in valid))
        if len(categories) < 2:
            return float('inf'), None
        
        n_total = len(y)
        weighted_gini = 0.0
        
        for cat in categories:
            cat_labels = [label for v, label in valid if v == cat]
            if cat_labels:
                weighted_gini += (len(cat_labels) / n_total) * self.gini(cat_labels)
        
        return weighted_gini, categories
    
    def best_split(self, X, y):
        best_feature = None
        best_gini = float('inf')
        best_info = None
        best_type = None
        
        n_features = len(X[0])
        
        for f in range(n_features):
            ftype = self.feature_types_.get(f, 'numeric')
            
            if ftype == 'numeric':
                g, threshold = self._best_numeric_split(X, y, f)
                if g < best_gini and threshold is not None:
                    best_gini = g
                    best_feature = f
                    best_info = threshold
                    best_type = 'numeric'
            else:
                g, categories = self._best_categorical_split(X, y, f)
                if g < best_gini and categories is not None:
                    best_gini = g
                    best_feature = f
                    best_info = categories
                    best_type = 'categorical'
        
        return best_feature, best_info, best_type
    
    def _split_numeric(self, X, y, feature, threshold):
        left_X, left_y = [], []
        right_X, right_y = [], []
        missing_X, missing_y = [], []
        
        for i in range(len(X)):
            val = X[i][feature]
            if val is None:
                missing_X.append(X[i])
                missing_y.append(y[i])
            elif val <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y, missing_X, missing_y
    
    def _split_categorical(self, X, y, feature, categories):
        partitions = {cat: ([], []) for cat in categories}
        missing_X, missing_y = [], []
        
        for i in range(len(X)):
            val = X[i][feature]
            if val is None:
                missing_X.append(X[i])
                missing_y.append(y[i])
            elif val in partitions:
                partitions[val][0].append(X[i])
                partitions[val][1].append(y[i])
        
        return partitions, missing_X, missing_y
    
    def build(self, X, y, depth=0):
        """Recursively build the tree."""
        class_probs = self._compute_class_probs(y)
        n_samples = len(y)
        
        if (len(set(y)) == 1 or 
            depth >= self.max_depth or 
            len(y) < self.min_samples_split):
            majority_class = Counter(y).most_common(1)[0][0]
            node = Node(value=majority_class, class_probs=class_probs)
            node.n_samples = n_samples
            return node
        
        feature, split_info, split_type = self.best_split(X, y)
        
        if feature is None:
            majority_class = Counter(y).most_common(1)[0][0]
            node = Node(value=majority_class, class_probs=class_probs)
            node.n_samples = n_samples
            return node
        
        if split_type == 'numeric':
            left_X, left_y, right_X, right_y, missing_X, missing_y = self._split_numeric(
                X, y, feature, split_info
            )
            
            left_X.extend(missing_X)
            left_y.extend(missing_y)
            right_X.extend(missing_X)
            right_y.extend(missing_y)
            
            if len(left_y) == 0 or len(right_y) == 0:
                majority_class = Counter(y).most_common(1)[0][0]
                node = Node(value=majority_class, class_probs=class_probs)
                node.n_samples = n_samples
                return node
            
            left = self.build(left_X, left_y, depth + 1)
            right = self.build(right_X, right_y, depth + 1)
            
            node = Node(feature=feature, threshold=split_info, left=left, right=right,
                       class_probs=class_probs, split_type='numeric')
            node.n_samples = n_samples
            
        else:
            partitions, missing_X, missing_y = self._split_categorical(
                X, y, feature, split_info
            )
            
            children = {}
            for cat, (cat_X, cat_y) in partitions.items():
                cat_X_extended = cat_X + missing_X
                cat_y_extended = cat_y + missing_y
                
                if len(cat_y_extended) > 0:
                    children[cat] = self.build(cat_X_extended, cat_y_extended, depth + 1)
            
            if not children:
                majority_class = Counter(y).most_common(1)[0][0]
                node = Node(value=majority_class, class_probs=class_probs)
                node.n_samples = n_samples
                return node
            
            node = Node(feature=feature, children=children, class_probs=class_probs,
                       split_type='categorical')
            node.n_samples = n_samples
        
        return node
    
    def fit(self, X, y, feature_types=None):
        self.classes_ = list(set(y))
        n_features = len(X[0]) if X else 0
        self.feature_types_ = feature_types or {i: 'numeric' for i in range(n_features)}
        self.root = self.build(X, y)
        return self
    
    def _predict_proba_one(self, x, node):
        if node.value is not None:
            return node.class_probs.copy()
        val = x[node.feature]
        
        if val is None:
            if node.split_type == 'numeric':
                branches = [node.left, node.right]
            else:
                branches = list(node.children.values())
            
            total_n = sum(b.n_samples for b in branches if b)
            if total_n == 0:
                return node.class_probs.copy()
            
            combined = {}
            for branch in branches:
                if branch is None:
                    continue
                w = branch.n_samples / total_n
                branch_probs = self._predict_proba_one(x, branch)
                for cls, prob in branch_probs.items():
                    combined[cls] = combined.get(cls, 0) + w * prob
            
            return combined
        
        if node.split_type == 'numeric':
            if val <= node.threshold:
                return self._predict_proba_one(x, node.left)
            else:
                return self._predict_proba_one(x, node.right)
        else:
            if val in node.children:
                return self._predict_proba_one(x, node.children[val])
            else:
                return node.class_probs.copy()
    
    def predict_one(self, x, node):
        probs = self._predict_proba_one(x, node)
        if not probs:
            return self.classes_[0] if self.classes_ else 0
        return max(probs, key=probs.get)
    
    def predict(self, X):
        return [self.predict_one(x, self.root) for x in X]
    
    def score(self, X, y):
        preds = self.predict(X)
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return correct / len(y)
    
    def _count_nodes(self, node):
        if node is None:
            return 0
        if node.value is not None:
            return 1
        
        if node.split_type == 'numeric':
            return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
        else:
            return 1 + sum(self._count_nodes(c) for c in node.children.values())
    
    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.value is not None:
            return 1
        
        if node.split_type == 'numeric':
            return self._count_leaves(node.left) + self._count_leaves(node.right)
        else:
            return sum(self._count_leaves(c) for c in node.children.values())
    
    def _get_depth(self, node):
        if node is None or node.value is not None:
            return 1
        
        if node.split_type == 'numeric':
            return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
        else:
            return 1 + max((self._get_depth(c) for c in node.children.values()), default=0)
    
    def get_stats(self):
        return {
            'nodes': self._count_nodes(self.root),
            'leaves': self._count_leaves(self.root),
            'depth': self._get_depth(self.root)
        }
    
    def _copy_node(self, node):
        """Deep copy a node."""
        if node is None:
            return None
        new_node = Node(
            feature=node.feature,
            threshold=node.threshold,
            value=node.value,
            class_probs=node.class_probs.copy() if node.class_probs else {},
            split_type=node.split_type
        )
        new_node.n_samples = node.n_samples
        new_node.left = self._copy_node(node.left)
        new_node.right = self._copy_node(node.right)
        new_node.children = {k: self._copy_node(v) for k, v in node.children.items()}
        return new_node
    
    def copy(self):
        """Create a copy of the tree."""
        new_tree = DecisionTree(self.max_depth, self.min_samples_split)
        new_tree.root = self._copy_node(self.root)
        new_tree.classes_ = self.classes_[:] if self.classes_ else None
        new_tree.feature_types_ = self.feature_types_.copy() if self.feature_types_ else None
        return new_tree
    
    def prune(self, X_val, y_val):
        """
        Post-pruning: Reduced Error Pruning (REP) 
    
        """
        
        def reaches_node(x, target_node, current_node=None):
            if current_node is None:
                current_node = self.root
            
            if current_node is target_node:
                return True
            
            if current_node.value is not None:
                return False
            
            val = x[current_node.feature]
            
            if val is None:
                # Check all branches
                if current_node.split_type == 'numeric':
                    return (reaches_node(x, target_node, current_node.left) or 
                            reaches_node(x, target_node, current_node.right))
                else:
                    return any(reaches_node(x, target_node, child) 
                              for child in current_node.children.values())
            
            if current_node.split_type == 'numeric':
                if val <= current_node.threshold:
                    return reaches_node(x, target_node, current_node.left)
                else:
                    return reaches_node(x, target_node, current_node.right)
            else:
                if val in current_node.children:
                    return reaches_node(x, target_node, current_node.children[val])
                return False
        
        def count_errors(X, y):
            if len(X) == 0:
                return 0
            preds = self.predict(X)
            return sum(1 for p, t in zip(preds, y) if p != t)
        
        def prune_node(node):
            if node is None or node.value is not None:
                return node
            
            if node.split_type == 'numeric':
                node.left = prune_node(node.left)
                node.right = prune_node(node.right)
            else:
                for key in node.children:
                    node.children[key] = prune_node(node.children[key])
            
            X_v, y_v = [], []
            for i, x in enumerate(X_val):
                if reaches_node(x, node):
                    X_v.append(x)
                    y_v.append(y_val[i])
            
            if len(X_v) == 0:
                majority = max(node.class_probs, key=node.class_probs.get) if node.class_probs else 0
                node.value = majority
                node.left = None
                node.right = None
                node.children = {}
                return node
            
            y_v_major = Counter(y_v).most_common(1)[0][0]
            
            errors_keep = count_errors(X_v, y_v)
            
            old_value = node.value
            old_left = node.left
            old_right = node.right
            old_children = node.children.copy()
            old_split_type = node.split_type
            
            node.value = y_v_major
            node.left = None
            node.right = None
            node.children = {}
            
            errors_leaf = count_errors(X_v, y_v)
            
            if errors_leaf <= errors_keep:
                return node
            else:
                node.value = old_value
                node.left = old_left
                node.right = old_right
                node.children = old_children
                node.split_type = old_split_type
                return node
        
        self.root = prune_node(self.root)
        return self

