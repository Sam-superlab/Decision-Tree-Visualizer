import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class NodeInfo:
    """Node information for visualization"""
    depth: int
    samples: int
    feature_idx: Optional[int]
    threshold: Optional[float]
    gini: float
    is_leaf: bool
    prediction: Optional[int]
    left_child: Optional['NodeInfo']
    right_child: Optional['NodeInfo']
    samples_indices: np.ndarray  # Store sample indices for visualization


class DecisionTreeVisualizer:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2,
                 criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.history: List[NodeInfo] = []
        self.root: Optional[NodeInfo] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train decision tree and record process"""
        self.history = []
        self.root = self._build_tree(
            X, y, depth=0, sample_indices=np.arange(len(y)))

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int,
                    sample_indices: np.ndarray) -> NodeInfo:
        """Recursively build decision tree"""
        n_samples = len(y)

        # Calculate current node impurity
        current_gini = self._calculate_gini(y) if self.criterion == 'gini' else \
            self._calculate_entropy(y)

        # Check termination conditions
        is_leaf = (depth >= self.max_depth or
                   n_samples < self.min_samples_split or
                   len(np.unique(y)) == 1)

        node = NodeInfo(
            depth=depth,
            samples=n_samples,
            feature_idx=None,
            threshold=None,
            gini=current_gini,
            is_leaf=is_leaf,
            prediction=np.argmax(np.bincount(y)) if is_leaf else None,
            left_child=None,
            right_child=None,
            samples_indices=sample_indices
        )

        # Record current node state
        self.history.append(node)

        if not is_leaf:
            # Find best split
            feature_idx, threshold, _ = self._find_best_split(X, y)
            node.feature_idx = feature_idx
            node.threshold = threshold

            # Split data
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            # Recursively build left and right subtrees
            left_indices = sample_indices[left_mask]
            right_indices = sample_indices[right_mask]

            node.left_child = self._build_tree(
                X[left_mask], y[left_mask], depth + 1, left_indices)
            node.right_child = self._build_tree(
                X[right_mask], y[right_mask], depth + 1, right_indices)

        return node

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy"""
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def get_history(self) -> List[dict]:
        """Convert training history to serializable format"""
        def serialize_node(node: NodeInfo) -> dict:
            node_dict = {
                'depth': node.depth,
                'samples': node.samples,
                'feature_idx': node.feature_idx,
                'threshold': float(node.threshold) if node.threshold is not None else None,
                'gini': float(node.gini),
                'is_leaf': node.is_leaf,
                'prediction': int(node.prediction) if node.prediction is not None else None,
                'samples_indices': node.samples_indices.tolist(),
                'children': []
            }

            if node.left_child:
                node_dict['children'].append(serialize_node(node.left_child))
            if node.right_child:
                node_dict['children'].append(serialize_node(node.right_child))

            return node_dict

        serialized_history = []
        for node in self.history:
            serialized_history.append(serialize_node(node))
        return serialized_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples"""
        def _predict_single(node: NodeInfo, x: np.ndarray) -> np.ndarray:
            if node.is_leaf:
                return node.prediction

            if x[node.feature_idx] <= node.threshold:
                return _predict_single(node.left_child, x)
            return _predict_single(node.right_child, x)

        return np.array([_predict_single(self.root, x) for x in X])

    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find best feature and threshold for splitting"""
        best_gini = float('inf')
        best_feature = -1
        best_threshold = 0.0

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                left_gini = self._calculate_gini(y[left_mask])
                right_gini = self._calculate_gini(y[right_mask])

                # Calculate weighted Gini impurity
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)

                weighted_gini = (n_left * left_gini +
                                 n_right * right_gini) / n_total

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini
