from flask import Flask, render_template, jsonify, request
from src.tree import DecisionTreeVisualizer
from src.visualizer import TreeVisualizer
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification

app = Flask(__name__)

DATASETS = {
    'moons': lambda: make_moons(n_samples=100, noise=0.25, random_state=42),
    'circles': lambda: make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=42),
    'linear': lambda: make_classification(n_samples=100, n_features=2, n_redundant=0,
                                          n_informative=2, random_state=42,
                                          n_clusters_per_class=1)
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/train')
def train():
    # 获取参数
    dataset_name = request.args.get('dataset', 'moons')
    max_depth = int(request.args.get('max_depth', 3))
    min_samples_split = int(request.args.get('min_samples_split', 2))
    criterion = request.args.get('criterion', 'gini')

    # 生成数据
    X, y = DATASETS[dataset_name]()

    # 训练决策树
    tree = DecisionTreeVisualizer(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion
    )
    tree.fit(X, y)

    # 获取训练历史
    history = tree.get_history()

    return jsonify({
        'status': 'success',
        'history': history,
        'data': {
            'X': X.tolist(),
            'y': y.tolist()
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
