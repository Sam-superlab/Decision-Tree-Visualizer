// Global variables
let currentStep = 0;
let history = [];
let isPlaying = false;
let playInterval = null;
const notyf = new Notyf();

// DOM elements
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

// Event listeners
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

// Train decision tree
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
            notyf.success('Decision tree training completed!');
        } else {
            throw new Error('Training failed');
        }
    } catch (error) {
        notyf.error('Error occurred during training');
        console.error(error);
    } finally {
        trainBtn.disabled = false;
    }
}

// Update visualization
function updateVisualization() {
    if (history.length === 0) return;

    const node = history[currentStep];

    // Update tree structure visualization
    updateTreeViz(node);

    // Update feature space visualization
    updateSpaceViz(node);
}

// Update tree structure visualization
function updateTreeViz(node) {
    // Prepare tree structure data
    const treeData = {
        nodes: [],
        edges: []
    };

    function processNode(node, parentId = null) {
        const nodeId = `node${treeData.nodes.length}`;

        // Create node label
        let label = `Depth: ${node.depth}<br>Samples: ${node.samples}<br>Gini: ${node.gini.toFixed(3)}`;
        if (node.is_leaf) {
            label += `<br>Prediction: ${node.prediction}`;
        } else {
            label += `<br>X[${node.feature_idx}] ≤ ${node.threshold.toFixed(2)}`;
        }

        // Add node
        treeData.nodes.push({
            id: nodeId,
            label: label,
            level: node.depth,
            color: node.is_leaf ? '#FF9999' : '#66B2FF'
        });

        // Add edge
        if (parentId !== null) {
            treeData.edges.push({
                from: parentId,
                to: nodeId
            });
        }

        // Process child nodes
        if (node.children) {
            node.children.forEach(child => {
                processNode(child, nodeId);
            });
        }
    }

    processNode(node);

    // Create tree graph
    const data = [{
        type: 'scatter',
        mode: 'markers+text',
        x: treeData.nodes.map((_, i) => i * 2),  // Horizontal node distribution
        y: treeData.nodes.map(n => -n.level * 2),  // Vertical node distribution
        text: treeData.nodes.map(n => n.label),
        textposition: 'top center',
        marker: {
            size: 20,
            color: treeData.nodes.map(n => n.color)
        },
        hoverinfo: 'text'
    }];

    // Add connecting lines
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

// Update feature space visualization
function updateSpaceViz(node) {
    // Create scatter plot and decision boundary using Plotly
    const data = [{
        type: 'scatter',
        x: node.X_values,
        y: node.y_values,
        mode: 'markers',
        marker: {
            color: node.colors,
            size: 10
        },
        name: 'Data Points'
    }];

    if (!node.is_leaf) {
        // Add split line
        if (node.feature_idx === 0) {
            data.push({
                type: 'line',
                x: [node.threshold, node.threshold],
                y: [-2, 2],
                line: {
                    color: 'red',
                    dash: 'dash'
                },
                name: 'Split Line'
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
                name: 'Split Line'
            });
        }
    }

    const layout = {
        title: `Depth: ${node.depth}, Gini: ${node.gini.toFixed(3)}`,
        xaxis: { range: [-2, 2] },
        yaxis: { range: [-2, 2] },
        width: document.getElementById('space-viz').clientWidth,
        height: document.getElementById('space-viz').clientHeight
    };

    Plotly.newPlot('space-viz', data, layout);
}

// Update control button states
function updateControls() {
    prevStepBtn.disabled = currentStep === 0;
    nextStepBtn.disabled = currentStep === history.length - 1;
    playPauseBtn.innerHTML = isPlaying ?
        '<i class="bi bi-pause-fill"></i> Pause' :
        '<i class="bi bi-play-fill"></i> Play';
}

// Show previous step
function showPreviousStep() {
    if (currentStep > 0) {
        currentStep--;
        updateVisualization();
        updateControls();
    }
}

// Show next step
function showNextStep() {
    if (currentStep < history.length - 1) {
        currentStep++;
        updateVisualization();
        updateControls();
    }
}

// Toggle play/pause state
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

// Redraw charts when window size changes
window.addEventListener('resize', () => {
    if (history.length > 0) {
        updateVisualization();
    }
}); 