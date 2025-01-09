// 全局变量
let currentStep = 0;
let history = [];
let isPlaying = false;
let playInterval = null;
const notyf = new Notyf();

// DOM元素
const maxDepthSlider = document.getElementById('max-depth');
const maxDepthValue = document.getElementById('max-depth-value');
const minSamplesSlider = document.getElementById('min-samples');
const minSamplesValue = document.getElementById('min-samples-value');
const datasetSelect = document.getElementById('dataset');
const criterionSelect = document.getElementById('criterion');
const trainBtn = document.getElementById('train-btn');
const prevStepBtn = document.getElementById('prev-step');
const nextStepBtn = document.getElementById('next-step');
const playPauseBtn = document.getElementById('play-pause');

// 事件监听器
maxDepthSlider.addEventListener('input', e => {
    maxDepthValue.textContent = e.target.value;
});

minSamplesSlider.addEventListener('input', e => {
    minSamplesValue.textContent = e.target.value;
});

trainBtn.addEventListener('click', trainTree);
prevStepBtn.addEventListener('click', showPreviousStep);
nextStepBtn.addEventListener('click', showNextStep);
playPauseBtn.addEventListener('click', togglePlayPause);

// 训练决策树
async function trainTree() {
    const params = new URLSearchParams({
        dataset: datasetSelect.value,
        max_depth: maxDepthSlider.value,
        min_samples_split: minSamplesSlider.value,
        criterion: criterionSelect.value
    });

    try {
        trainBtn.disabled = true;
        const response = await fetch(`/api/train?${params}`);
        const data = await response.json();

        if (data.status === 'success') {
            history = data.history;
            currentStep = 0;
            updateVisualization();
            updateControls();
            notyf.success('决策树训练完成！');
        } else {
            throw new Error('训练失败');
        }
    } catch (error) {
        notyf.error('训练过程中发生错误');
        console.error(error);
    } finally {
        trainBtn.disabled = false;
    }
}

// 更新可视化
function updateVisualization() {
    if (history.length === 0) return;

    const node = history[currentStep];

    // 更新树结构可视化
    updateTreeViz(node);

    // 更新特征空间可视化
    updateSpaceViz(node);
}

// 更新树结构可视化
function updateTreeViz(node) {
    // 准备树结构数据
    const treeData = {
        nodes: [],
        edges: []
    };

    function processNode(node, parentId = null) {
        const nodeId = `node${treeData.nodes.length}`;

        // 创建节点标签
        let label = `深度: ${node.depth}<br>样本数: ${node.samples}<br>基尼: ${node.gini.toFixed(3)}`;
        if (node.is_leaf) {
            label += `<br>预测: ${node.prediction}`;
        } else {
            label += `<br>X[${node.feature_idx}] ≤ ${node.threshold.toFixed(2)}`;
        }

        // 添加节点
        treeData.nodes.push({
            id: nodeId,
            label: label,
            level: node.depth,
            color: node.is_leaf ? '#FF9999' : '#66B2FF'
        });

        // 添加边
        if (parentId !== null) {
            treeData.edges.push({
                from: parentId,
                to: nodeId
            });
        }

        // 处理子节点
        if (node.children) {
            node.children.forEach(child => {
                processNode(child, nodeId);
            });
        }
    }

    processNode(node);

    // 创建树图
    const data = [{
        type: 'scatter',
        mode: 'markers+text',
        x: treeData.nodes.map((_, i) => i * 2),  // 水平分布节点
        y: treeData.nodes.map(n => -n.level * 2),  // 垂直分布节点
        text: treeData.nodes.map(n => n.label),
        textposition: 'top center',
        marker: {
            size: 20,
            color: treeData.nodes.map(n => n.color)
        },
        hoverinfo: 'text'
    }];

    // 添加连接线
    treeData.edges.forEach(edge => {
        const fromNode = treeData.nodes[parseInt(edge.from.replace('node', ''))];
        const toNode = treeData.nodes[parseInt(edge.to.replace('node', ''))];
        const fromX = parseInt(edge.from.replace('node', '')) * 2;
        const toX = parseInt(edge.to.replace('node', '')) * 2;
        const fromY = -fromNode.level * 2;
        const toY = -toNode.level * 2;

        data.push({
            type: 'scatter',
            mode: 'lines',
            x: [fromX, toX],
            y: [fromY, toY],
            line: {
                color: 'gray',
                width: 1
            },
            showlegend: false,
            hoverinfo: 'none'
        });
    });

    const layout = {
        showlegend: false,
        hovermode: 'closest',
        margin: { t: 40, l: 40, r: 40, b: 40 },
        xaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        yaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        width: document.getElementById('tree-viz').clientWidth,
        height: document.getElementById('tree-viz').clientHeight
    };

    Plotly.newPlot('tree-viz', data, layout);
}

// 更新特征空间可视化
function updateSpaceViz(node) {
    // 使用Plotly创建散点图和决策边界
    const data = [{
        type: 'scatter',
        x: node.X_values,
        y: node.y_values,
        mode: 'markers',
        marker: {
            color: node.colors,
            size: 10
        },
        name: '数据点'
    }];

    if (!node.is_leaf) {
        // 添加分割线
        if (node.feature_idx === 0) {
            data.push({
                type: 'line',
                x: [node.threshold, node.threshold],
                y: [-2, 2],
                line: {
                    color: 'red',
                    dash: 'dash'
                },
                name: '分割线'
            });
        } else {
            data.push({
                type: 'line',
                x: [-2, 2],
                y: [node.threshold, node.threshold],
                line: {
                    color: 'red',
                    dash: 'dash'
                },
                name: '分割线'
            });
        }
    }

    const layout = {
        title: `深度: ${node.depth}, 基尼: ${node.gini.toFixed(3)}`,
        xaxis: { range: [-2, 2] },
        yaxis: { range: [-2, 2] },
        width: document.getElementById('space-viz').clientWidth,
        height: document.getElementById('space-viz').clientHeight
    };

    Plotly.newPlot('space-viz', data, layout);
}

// 更新控制按钮状态
function updateControls() {
    prevStepBtn.disabled = currentStep === 0;
    nextStepBtn.disabled = currentStep === history.length - 1;
    playPauseBtn.innerHTML = isPlaying ?
        '<i class="bi bi-pause-fill"></i> 暂停' :
        '<i class="bi bi-play-fill"></i> 播放';
}

// 显示上一步
function showPreviousStep() {
    if (currentStep > 0) {
        currentStep--;
        updateVisualization();
        updateControls();
    }
}

// 显示下一步
function showNextStep() {
    if (currentStep < history.length - 1) {
        currentStep++;
        updateVisualization();
        updateControls();
    }
}

// 切换播放/暂停状态
function togglePlayPause() {
    isPlaying = !isPlaying;
    if (isPlaying) {
        playInterval = setInterval(() => {
            if (currentStep < history.length - 1) {
                showNextStep();
            } else {
                togglePlayPause();
            }
        }, 1000);
    } else {
        clearInterval(playInterval);
    }
    updateControls();
}

// 窗口大小改变时重新绘制图表
window.addEventListener('resize', () => {
    if (history.length > 0) {
        updateVisualization();
    }
}); 