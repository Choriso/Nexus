let simulation;
let svg, g, link, node;
let editingNodeId = null; // string или null

const COLORS = {
    'core':       '#ff8906',
    'work':       '#2cb67d',
    'hobby':      '#7f5af0',
    'psychology': '#3da9fc',
    'default':    '#72757e'
};

const CAT_LABELS = {
    'work':       'Работа',
    'hobby':      'Хобби',
    'psychology': 'Психология'
};

/**
 * Получить цвет узла по его категории.
 *
 * @param {Object} d - Узел графа.
 * @returns {string} Цвет hex.
 */
function nodeColor(d) {
    return COLORS[d.category] || COLORS['default'];
}

/**
 * Получить радиус узла, зависящий от типа узла.
 *
 * @param {Object} d - Узел графа.
 * @returns {number} Радиус узла в пикселях.
 */
function nodeRadius(d) {
    if (d.category === 'core') return 18;
    if (d.is_branch) return 13;
    return 9;
}

/**
 * Инициализация и отрисовка графа в контейнере #graphCanvas.
 *
 * @param {Array<Object>} rawNodes - Список узлов, полученных от сервера.
 * @param {Array<Object>} rawConnections - Список связей, полученных от сервера.
 * @param {string} userName - Имя пользователя для отображения в центральном узле.
 * @returns {void}
 */
window.initGraph = function(rawNodes, rawConnections, userName) {
    const canvas = document.getElementById('graphCanvas');
    if (!canvas) { console.error('graphCanvas not found'); return; }

    const width  = canvas.clientWidth  || 900;
    const height = canvas.clientHeight || 650;

    const coreNode = {
        id: 'user_core',
        title: userName || 'Я',
        category: 'core',
        x: width / 2, y: height / 2,
        fx: width / 2, fy: height / 2
    };

    const finalNodes = [coreNode];
    const finalLinks = [];

    Object.entries(CAT_LABELS).forEach(([cat, label]) => {
        finalNodes.push({ id: `branch_${cat}`, title: label, category: cat, is_branch: true });
        finalLinks.push({ source: 'user_core', target: `branch_${cat}` });
    });

    const nodeIdSet = new Set(finalNodes.map(n => String(n.id)));

    rawNodes.forEach(n => {
        const nodeId = String(n.id);
        if (nodeId === 'user_core' || n.category === 'core') return;

        let cat = String(n.category || 'hobby').toLowerCase().trim();
        if (!CAT_LABELS[cat]) cat = 'hobby';

        finalNodes.push({
            ...n,
            id: nodeId,
            category: cat,
            x: n.x || undefined,
            y: n.y || undefined,
        });
        nodeIdSet.add(nodeId);

        finalLinks.push({ source: `branch_${cat}`, target: nodeId });
    });

    (rawConnections || []).forEach(c => {
        const sourceId = String(c.from ?? c.from_node_id ?? c.source ?? '');
        const targetId = String(c.to ?? c.to_node_id ?? c.target ?? '');
        if (!sourceId || !targetId) return;
        if (!nodeIdSet.has(sourceId) || !nodeIdSet.has(targetId)) return;

        finalLinks.push({
            id: String(c.id ?? `${sourceId}_${targetId}`),
            source: sourceId,
            target: targetId,
            label: c.label || ''
        });
    });

    svg = d3.select('#graphCanvas');
    svg.selectAll('*').remove();

    g = svg.append('g');

    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', e => g.attr('transform', e.transform));
    svg.call(zoom).on('dblclick.zoom', null);

    simulation = d3.forceSimulation(finalNodes)
        .force('link',      d3.forceLink(finalLinks).id(d => d.id).distance(90))
        .force('charge',    d3.forceManyBody().strength(-250))
        .force('center',    d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 8));

    link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(finalLinks)
        .join('line')
        .attr('stroke', '#444b5a')
        .attr('stroke-opacity', 0.45)
        .attr('stroke-width', 1.5);

    node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('.node')
        .data(finalNodes)
        .join('g')
        .attr('class', 'node')
        .call(
            d3.drag()
                .on('start', (e, d) => {
                    if (!e.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on('drag', (e, d) => {
                    d.fx = e.x; d.fy = e.y;
                })
                .on('end', (e, d) => {
                    if (!e.active) simulation.alphaTarget(0);
                    if (d.category !== 'core') {
                        _saveNodePosition(d.id, d.x, d.y);
                        d.fx = null; d.fy = null;
                    }
                })
        );

    node.append('circle')
        .attr('r',            nodeRadius)
        .attr('fill',         nodeColor)
        .attr('stroke',       '#16161a')
        .attr('stroke-width', 2);

    node.append('text')
        .attr('dy', d => -nodeRadius(d) - 6)
        .attr('text-anchor', 'middle')
        .style('fill',      '#94a1b2')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('user-select',    'none')
        .text(d => d.title);

    node.on('click', (event, d) => {
        if (d.category === 'core' || d.is_branch) return;
        d3.selectAll('circle').attr('stroke', '#16161a').attr('stroke-width', 2);
        d3.select(event.currentTarget).select('circle')
            .attr('stroke', '#7f5af0').attr('stroke-width', 3);

        if (typeof window.onNodeClick === 'function') window.onNodeClick(d.id);
    });

    node.on('contextmenu', (event, d) => {
        event.preventDefault();
        if (d.category === 'core' || d.is_branch) return;
        window.openEditModal(d);
    });

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });
};

/**
 * Добавить новый пользовательский узел в граф.
 *
 * @returns {Promise<void>}
 */
window.addNode = async function() {
    const title = prompt('Название нового узла:');
    if (!title) return;

    try {
        const res = await fetch('/knowledge_graph/node', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, category: 'hobby', description: '' })
        });
        const data = await res.json();
        if (data.success) {
            await _reloadGraph();
        } else {
            alert('Ошибка: ' + (data.message || 'неизвестная'));
        }
    } catch (err) {
        console.error('[addNode]', err);
    }
};

/**
 * Сохранить текущие позиции всех пользовательских узлов.
 *
 * @returns {Promise<void>}
 */
window.saveGraphPositions = async function() {
    if (!simulation) return;

    const positions = simulation.nodes()
        .filter(d => d.category !== 'core' && !d.is_branch)
        .map(d => ({ id: d.id, x: Math.round(d.x), y: Math.round(d.y) }));

    for (const pos of positions) {
        await _saveNodePosition(pos.id, pos.x, pos.y);
    }
    _showToast('Позиции сохранены ✓');
};

/**
 * Сохранить позицию одного узла на сервере.
 *
 * @param {string} nodeId - Идентификатор узла.
 * @param {number} x - Координата X.
 * @param {number} y - Координата Y.
 * @returns {Promise<void>}
 */
async function _saveNodePosition(nodeId, x, y) {
    if (typeof nodeId === 'string' && nodeId.startsWith('branch_')) return;
    if (nodeId === 'user_core') return;

    try {
        await fetch(`/knowledge_graph/node/${nodeId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: Math.round(x), y: Math.round(y) })
        });
    } catch (err) {
        console.warn('[savePos]', err);
    }
}

/**
 * Открыть модальное окно для редактирования узла.
 *
 * @param {Object} nodeData - Данные узла для редактирования.
 * @returns {void}
 */
window.openEditModal = function(nodeData) {
    editingNodeId = String(nodeData.id);

    document.getElementById('nodeTitle').value       = nodeData.title       || '';
    document.getElementById('nodeDescription').value = nodeData.description || '';
    document.getElementById('nodeCategory').value    = nodeData.category    || '';

    const modal = document.getElementById('nodeModal');
    modal.style.display = 'flex';
};

/**
 * Закрыть модальное окно редактирования и сбросить состояние.
 *
 * @returns {void}
 */
window.closeModal = function() {
    editingNodeId = null;
    document.getElementById('nodeModal').style.display = 'none';
};

/**
 * Сохранить изменения узла после редактирования.
 *
 * @returns {Promise<void>}
 */
window.saveNode = async function() {
    if (!editingNodeId) return;

    const payload = {
        title:       document.getElementById('nodeTitle').value,
        description: document.getElementById('nodeDescription').value,
        category:    document.getElementById('nodeCategory').value,
    };

    try {
        const res = await fetch(`/knowledge_graph/node/${editingNodeId}`, {
            method:  'PUT',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.success) {
            window.closeModal();
            await _reloadGraph();
        } else {
            alert('Ошибка сохранения: ' + (data.message || '?'));
        }
    } catch (err) {
        console.error('[saveNode]', err);
    }
};

/**
 * Удалить выбранный пользовательский узел из графа.
 *
 * @returns {Promise<void>}
 */
window.deleteNode = async function() {
    if (!editingNodeId) return;
    if (!confirm('Удалить узел?')) return;

    try {
        const res = await fetch(`/knowledge_graph/node/${editingNodeId}`, { method: 'DELETE' });
        const data = await res.json();

        if (data.success) {
            window.closeModal();
            await _reloadGraph();
        } else {
            alert('Ошибка удаления: ' + (data.message || '?'));
        }
    } catch (err) {
        console.error('[deleteNode]', err);
    }
};

/**
 * Обработчик клика по фону модального окна для закрытия.
 *
 * @param {Event} e - Событие клика.
 * @returns {void}
 */
window.addEventListener('click', e => {
    const modal = document.getElementById('nodeModal');
    if (modal && e.target === modal) window.closeModal();
});

/**
 * Перезагрузка данных графа с сервера и повторная инициализация графа.
 *
 * @returns {Promise<void>}
 */
async function _reloadGraph() {
    try {
        const res  = await fetch('/knowledge_graph_data');
        const data = await res.json();

        const nodes       = data.nodes.map(n => ({ ...n, id: String(n.id) }));
        const connections = data.connections || [];

        const userName = document.getElementById('graphCanvas')?.dataset.username || '';
        window.initGraph(nodes, connections, userName);
    } catch (err) {
        console.error('[reloadGraph]', err);
    }
}

/**
 * Показывает всплывающее toast-уведомление.
 *
 * @param {string} msg - Текст уведомления.
 * @returns {void}
 */
function _showToast(msg) {
    let toast = document.getElementById('_graphToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = '_graphToast';
        Object.assign(toast.style, {
            position: 'fixed', bottom: '24px', left: '50%',
            transform: 'translateX(-50%)',
            background: '#2cb67d', color: '#fff',
            padding: '10px 20px', borderRadius: '8px',
            fontWeight: '600', zIndex: '9999',
            transition: 'opacity 0.3s'
        });
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 2500);
}