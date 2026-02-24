"""
Decision Tree Implementation with ID3 algorithm and Gini criterion.
Supports missing values via proportional routing and cost-complexity pruning.
"""

import numpy as np
from pandas import DataFrame
from typing import Optional, Tuple, List, Union


def _gini(y: np.ndarray) -> float:
    """
    Calculate Gini impurity for a set of labels.
    Args:
        y: Array of labels
    Returns:
        Gini impurity (0 for pure, 0.5 for balanced binary, 1 for one-class with binary convention)
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)


class TreeNode:
    """
    Node in a decision tree.

    Attributes:
        feature_idx: Index of feature used for split (None for leaf)
        threshold: Threshold value for the split (None for leaf)
        left: Left child (feature <= threshold)
        right: Right child (feature > threshold)
        value: Predicted class label if leaf
        proba: Probability of positive class if leaf
        is_leaf: Whether this node is a leaf
        left_weight: Proportion of samples (with non-missing feature) that go left
        right_weight: Proportion of samples (with non-missing feature) that go right
        n_left: Number of training samples that went left (with feature <= threshold)
        n_right: Number of training samples that went right (with feature > threshold)
    """

    def __init__(self, value: Optional[float] = None, proba: Optional[float] = None,
                 feature_idx: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Class label for leaf
        self.proba = proba  # Probability for class 1
        self.is_leaf = value is not None
        self.left_weight = 1.0  # Default: all to left if not set
        self.right_weight = 0.0
        self.n_left = 0
        self.n_right = 0
        # Training statistics (for pruning)
        self.n_train_samples = 0  # Number of training samples that reached this node
        self.train_majority = None  # Majority class from training
        self.train_proba = None  # Probability of class 1 from training

    def predict_single(self, x: np.ndarray) -> float:
        """
        Predict class label for a single sample.
        Args:
            x: Feature vector (1D array)
        Returns:
            Class label (-1 or 1)
        """
        if self.is_leaf:
            return self.value

        feature_val = x[self.feature_idx]

        # Handle missing values (NaN)
        if np.isnan(feature_val):
            # Recursively get predictions from both children with weights
            left_pred = self.left.predict_single(x) if self.left else self.proba
            right_pred = self.right.predict_single(x) if self.right else self.proba

            # If both children exist, combine
            if self.left and self.right:
                return self.left_weight * left_pred + self.right_weight * right_pred
            # If only one child exists (shouldn't happen normally), use that
            elif self.left:
                return left_pred
            elif self.right:
                return right_pred
            else:
                return self.proba

        # Feature is present, follow appropriate branch
        if feature_val <= self.threshold:
            return self.left.predict_single(x) if self.left else self.proba
        else:
            return self.right.predict_single(x) if self.right else self.proba

    def predict_proba_single(self, x: np.ndarray) -> float:
        """
        Predict probability of positive class for a single sample.
        Args:
            x: Feature vector (1D array)
        Returns:
            Probability of class 1 (between 0 and 1)
        """
        if self.is_leaf:
            return self.proba

        feature_val = x[self.feature_idx]

        # Handle missing values (NaN)
        if np.isnan(feature_val):
            left_proba = self.left.predict_proba_single(x) if self.left else self.proba
            right_proba = self.right.predict_proba_single(x) if self.right else self.proba

            if self.left and self.right:
                return self.left_weight * left_proba + self.right_weight * right_proba
            elif self.left:
                return left_proba
            elif self.right:
                return right_proba
            else:
                return self.proba

        if feature_val <= self.threshold:
            return self.left.predict_proba_single(x) if self.left else self.proba
        else:
            return self.right.predict_proba_single(x) if self.right else self.proba


class DecisionTree:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2, random_seed: Optional[int] = None):
        """
        Args:
            max_depth: Maximum depth of the tree (None for unlimited)
            min_samples_split: Minimum number of samples required to split a node
            random_seed: Random seed for reproducibility (for missing value handling)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.root: Optional[TreeNode] = None
        self.n_classes_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, DataFrame], y: np.ndarray,
        X_val: Optional[Union[np.ndarray, DataFrame]] = None,
        y_val: Optional[np.ndarray] = None,
        prune: bool = False
    ):
        """
        Build decision tree from training data.
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,), values should be -1 or 1
            X_val: Validation features for pruning (optional)
            y_val: Validation labels for pruning (optional)
            prune: Whether to prune the tree after training using validation set
        Returns:
            self
        """
        # Ensure y is in correct format
        unique_classes = np.unique(y)
        self.n_classes_ = len(unique_classes)

        if isinstance(X, DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.to_numpy()
        if isinstance(X_val, DataFrame):
            X_val = X_val.to_numpy()

        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0)

        # Calculate feature importances based on total Gini gain
        self._calculate_feature_importances(X, y)

        # Prune if requested and validation data provided
        if prune and X_val is not None and y_val is not None:
            self._prune_tree(X_val, y_val)

        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        """
        Recursively build the decision tree.
        Args:
            X: Feature matrix for current node
            y: Labels for current node
            depth: Current depth in the tree
        Returns:
            TreeNode instance (either leaf or internal node)
        """
        n_samples, n_features = X.shape

        # Compute training statistics for this node (used for pruning)
        majority_class = self._majority_class(y)
        proba = np.mean(y == 1)

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            # Create leaf node
            node = TreeNode(value=majority_class, proba=proba)
            node.n_train_samples = n_samples
            node.train_majority = majority_class
            node.train_proba = proba
            return node

        # Find best split
        best_feature, best_threshold, left_mask, right_mask = DecisionTree._best_split(X, y)

        # If no valid split found, create leaf
        if best_feature is None:
            node = TreeNode(value=majority_class, proba=proba)
            node.n_train_samples = n_samples
            node.train_majority = majority_class
            node.train_proba = proba
            return node

        # Create child nodes recursively
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]

        # Check if split produces valid children
        if len(left_indices) == 0 or len(right_indices) == 0:
            node = TreeNode(value=majority_class, proba=proba)
            node.n_train_samples = n_samples
            node.train_majority = majority_class
            node.train_proba = proba
            return node

        # Compute weights for missing value handling
        left_weight = len(left_indices) / (len(left_indices) + len(right_indices))
        right_weight = len(right_indices) / (len(left_indices) + len(right_indices))

        # Build left and right subtrees
        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Create current internal node
        node = TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )
        node.left_weight = left_weight
        node.right_weight = right_weight
        node.n_left = len(left_indices)
        node.n_right = len(right_indices)

        # Set training statistics for this internal node
        node.n_train_samples = n_samples
        node.train_majority = majority_class
        node.train_proba = proba

        return node

    @staticmethod
    def _best_split(X: np.ndarray, y: np.ndarray) -> Tuple[
        Optional[int], Optional[float], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Find the best split for the current node.
        Args:
            X: Feature matrix
            y: Labels
        Returns:
            Tuple of (best_feature_idx, best_threshold, left_mask, right_mask)
            Returns (None, None, None, None) if no valid split found
        """
        n_samples, n_features = X.shape
        current_gini = _gini(y)
        best_gini = current_gini
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None

        for feature_idx in range(n_features):
            # Get feature column
            feature_values = X[:, feature_idx]

            # Identify non-missing samples
            non_missing_mask = ~np.isnan(feature_values)
            if np.sum(non_missing_mask) == 0:
                # All values missing, skip this feature
                continue

            feature_valid = feature_values[non_missing_mask]
            y_valid = y[non_missing_mask]

            # Get unique sorted values
            unique_values = np.unique(feature_valid)
            if len(unique_values) <= 1:
                # Only one unique value, can't split
                continue

            # For binary features (0/1 after one-hot encoding), only test threshold 0.5
            if len(unique_values) == 2 and set(unique_values) == {0.0, 1.0}:
                thresholds = [0.5]
            else:
                # For continuous features, test midpoints between consecutive unique values
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            # Evaluate each threshold
            for threshold in thresholds:
                left_mask_full = np.zeros(n_samples, dtype=bool)
                right_mask_full = np.zeros(n_samples, dtype=bool)

                # Non-missing samples: assign based on threshold
                left_mask_valid = feature_valid <= threshold
                right_mask_valid = feature_valid > threshold

                left_mask_full[non_missing_mask] = left_mask_valid
                right_mask_full[non_missing_mask] = right_mask_valid

                # Missing samples: distributed to both with equal weight during training
                # For split evaluation, we ignore them (they don't contribute to Gini calculation)
                # But we still count them in the split for future prediction weights

                n_left = np.sum(left_mask_full)
                n_right = np.sum(right_mask_full)

                if n_left == 0 or n_right == 0:
                    continue

                # Calculate weighted Gini
                total = n_left + n_right
                gini_left = _gini(y_valid[left_mask_valid])
                gini_right = _gini(y_valid[right_mask_valid])
                weighted_gini = (n_left / total) * gini_left + (n_right / total) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_mask = left_mask_full
                    best_right_mask = right_mask_full

        return best_feature, best_threshold, best_left_mask, best_right_mask

    def _majority_class(self, y: np.ndarray) -> float:
        """
        Get the majority class in a set of labels.
        For tie, randomly choose (with consistent random state).
        Args:
            y: Array of labels
        Returns:
            Majority class label (-1 or 1)
        """
        unique, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        max_classes = unique[counts == max_count]
        if len(max_classes) == 1:
            return max_classes[0]
        elif self.rng is not None:
            return self.rng.choice(max_classes)
        else:
            return max_classes[0]

    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """
        Calculate feature importances based on total Gini reduction across the tree.
        Args:
            X: Training features
            y: Training labels
        """
        n_features = X.shape[1]
        total_importances = np.zeros(n_features)

        # Alternative: recompute by passing data through tree
        def compute_importance(node: TreeNode, X_node: np.ndarray, y_node: np.ndarray):
            if node.is_leaf:
                return

            # Skip if no valid split
            if node.feature_idx is None or node.threshold is None:
                return

            feature_idx = node.feature_idx
            threshold = node.threshold

            # Get non-missing samples
            non_missing_mask = ~np.isnan(X_node[:, feature_idx])
            X_valid = X_node[non_missing_mask]
            y_valid = y_node[non_missing_mask]

            if len(y_valid) == 0:
                return

            feature_valid = X_valid[:, feature_idx]
            left_mask = feature_valid <= threshold
            right_mask = feature_valid > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                return

            # Parent Gini
            parent_gini = _gini(y_valid)

            # Children Gini
            left_gini = _gini(y_valid[left_mask])
            right_gini = _gini(y_valid[right_mask])

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            total = n_left + n_right

            # Weighted Gini after split
            child_gini = (n_left / total) * left_gini + (n_right / total) * right_gini

            # Importance = Gini reduction * samples at this node
            importance = (parent_gini - child_gini) * total
            total_importances[feature_idx] += importance

            # Recurse on children using the valid subsets
            X_left = X_valid[left_mask] if np.any(left_mask) else None
            y_left = y_valid[left_mask] if np.any(left_mask) else None
            X_right = X_valid[right_mask] if np.any(right_mask) else None
            y_right = y_valid[right_mask] if np.any(right_mask) else None

            if X_left is not None and len(X_left) > 0:
                compute_importance(node.left, X_left, y_left)
            if X_right is not None and len(X_right) > 0:
                compute_importance(node.right, X_right, y_right)

        compute_importance(self.root, X, y)

        # Normalize importances to sum to 1
        if np.sum(total_importances) > 0:
            self.feature_importances_ = total_importances / np.sum(total_importances)
        else:
            self.feature_importances_ = total_importances

    def predict(self, X: Union[np.ndarray, DataFrame]) -> np.ndarray:
        """
        Predict class labels for samples.
        Args:
            X: Feature matrix, shape (n_samples, n_features)
        Returns:
            Array of predicted labels (-1 or 1)
        """
        if self.root is None:
            raise ValueError("Tree hasn't been fitted yet")

        if isinstance(X, DataFrame):
            if self.feature_names_:
                X = X[self.feature_names_].to_numpy()
            else:
                X = X.to_numpy()

        predictions = np.array([self.root.predict_single(x) for x in X])
        return predictions

    def predict_proba(self, X: Union[np.ndarray, DataFrame]) -> np.ndarray:
        """
        Predict probabilities for the positive class (label 1).
        Args:
            X: Feature matrix, shape (n_samples, n_features)
        Returns:
            Array of probabilities for class 1
        """
        if self.root is None:
            raise ValueError("Tree hasn't been fitted yet")

        if isinstance(X, DataFrame):
            if self.feature_names_:
                X = X[self.feature_names_].to_numpy()
            else:
                X = X.to_numpy()

        probabilities = np.array([self.root.predict_proba_single(x) for x in X])
        return probabilities

    def _prune_tree(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Prune the tree using reduced-error pruning on validation set.
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        if self.root is None:
            return

        # Collect all internal nodes in post-order (bottom-up)
        internal_nodes = []

        def collect_internal_nodes(_node: Optional[TreeNode]):
            if _node is None or _node.is_leaf:
                return
            # Recurse first to children (post-order)
            collect_internal_nodes(_node.left)
            collect_internal_nodes(_node.right)
            internal_nodes.append(_node)

        collect_internal_nodes(self.root)

        # Current accuracy (will be updated greedily)
        y_pred_current = self.predict(X_val)
        current_accuracy = np.mean(y_pred_current == y_val)

        # Process each internal node in bottom-up order
        for node in internal_nodes:
            if node.is_leaf:
                continue  # Already leaf

            # Save children to restore if pruning not beneficial
            original_left = node.left
            original_right = node.right
            original_is_leaf = node.is_leaf
            original_value = node.value
            original_proba = node.proba

            # Temporarily convert node to leaf using training majority
            node.is_leaf = True
            node.value = node.train_majority
            node.proba = node.train_proba
            node.left = None
            node.right = None

            # Compute accuracy on validation set with this temporary change
            y_pred_trial = self.predict(X_val)
            trial_accuracy = np.mean(y_pred_trial == y_val)

            # If accuracy does not degrade (or improves), keep this node as leaf (permanently)
            # Use a small tolerance to account for floating point
            if trial_accuracy >= current_accuracy - 1e-6:
                # Keep leaf status; update current accuracy to the trial accuracy
                current_accuracy = trial_accuracy
            else:
                # Revert to original internal node
                node.is_leaf = original_is_leaf
                node.value = original_value
                node.proba = original_proba
                node.left = original_left
                node.right = original_right
                # current_accuracy remains unchanged

    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        def depth(node: Optional[TreeNode]) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        return depth(self.root)

    def get_n_nodes(self) -> int:
        """Get total number of nodes in the tree."""
        def count(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)
        return count(self.root)

    def get_n_leaves(self) -> int:
        """Get number of leaf nodes."""
        def count_leaves(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return count_leaves(node.left) + count_leaves(node.right)
        return count_leaves(self.root)
