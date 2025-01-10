import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from matplotlib.colors import ListedColormap


class TreeVisualizer:
    def __init__(self):
        self.fig = None
        self.ax_tree = None
        self.ax_space = None
        self.cmap = ListedColormap(['#FF9999', '#66B2FF'])

    def plot_step(self, X: np.ndarray, y: np.ndarray, node_info: 'NodeInfo',
                  show_decision_boundary: bool = True) -> None:
        """Plot one step of decision tree growth"""
        if self.fig is None:
            self.fig, (self.ax_tree, self.ax_space) = plt.subplots(
                1, 2, figsize=(15, 6))

        self._plot_tree_structure(node_info)
        self._plot_feature_space(X, y, node_info, show_decision_boundary)
        plt.tight_layout()

    def _plot_tree_structure(self, node_info: 'NodeInfo') -> None:
        """Draw tree structure using networkx"""
        self.ax_tree.clear()
        G = nx.DiGraph()

        def add_nodes_edges(node: Optional['NodeInfo'], parent_id: Optional[str] = None):
            if node is None:
                return

            node_id = f"d{node.depth}_s{node.samples}"

            # Add node label
            label = f"samples={node.samples}\n"
            label += f"gini={node.gini:.3f}\n"
            if node.is_leaf:
                label += f"pred={node.prediction}"
            else:
                label += f"X[{node.feature_idx}] <= {node.threshold:.2f}"

            G.add_node(node_id, label=label)

            if parent_id:
                G.add_edge(parent_id, node_id)

            if not node.is_leaf:
                add_nodes_edges(node.left_child, node_id)
                add_nodes_edges(node.right_child, node_id)

        add_nodes_edges(node_info)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=self.ax_tree, with_labels=True,
                node_color='lightblue', node_size=2000)

        # Add node labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, ax=self.ax_tree)

    def _plot_feature_space(self, X: np.ndarray, y: np.ndarray,
                            node_info: 'NodeInfo',
                            show_decision_boundary: bool = True) -> None:
        """Draw feature space partitioning"""
        self.ax_space.clear()

        # Draw scatter plot
        scatter = self.ax_space.scatter(X[:, 0], X[:, 1], c=y,
                                        cmap=self.cmap, alpha=0.6)

        if show_decision_boundary:
            # Create grid for decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))

            # Get current node prediction
            if not node_info.is_leaf:
                if node_info.feature_idx == 0:
                    Z = xx <= node_info.threshold
                else:
                    Z = yy <= node_info.threshold

                # Draw decision boundary
                self.ax_space.contourf(xx, yy, Z, alpha=0.2, cmap=self.cmap)

                # Draw split line
                if node_info.feature_idx == 0:
                    self.ax_space.axvline(x=node_info.threshold,
                                          color='r', linestyle='--')
                else:
                    self.ax_space.axhline(y=node_info.threshold,
                                          color='r', linestyle='--')

        # Set plot properties
        self.ax_space.set_title(
            f'Depth: {node_info.depth}, Samples: {node_info.samples}\n'
            f'Gini: {node_info.gini:.3f}')
        self.ax_space.set_xlabel('Feature 0')
        self.ax_space.set_ylabel('Feature 1')

        # Add legend
        legend1 = self.ax_space.legend(*scatter.legend_elements(),
                                       loc="upper right", title="Classes")
        self.ax_space.add_artist(legend1)
