/**
 * Graph Editor for Knowledge Base
 */

let simulation;
let svg, g, link, node;
let editingNodeId = null;

const colors = {
    'core': '#ff8906',
    'work': '#2cb67d',
    'hobby': '#7f5af0',
    'psychology': '#3da9fc',
    'default': '#72757e'
};

// Экспортируем функцию в глобальную видимость сразу
window.initGraph = function(rawNodes, rawConnections) {
    const canvas = document.getElementById('graphCanvas');
    if (!canvas) return;
    const width = canvas.clientWidth;
    const height = 600;

    let finalNodes = [...rawNodes];
    let finalLinks = [];

    let coreNode = finalNodes.find(n => n.category === 'core') || finalNodes[0];

    const catTitles = {
        'work': 'Работа',
        'hobby': 'Хобби',
        'psychology': 'Психология',
        'default': 'Разное'
    };

    const categories = [...new Set(finalNodes.map(n => n.category))].filter(c => c && c !== 'core');

    categories.forEach(cat => {
        let parentNode = finalNodes.find(n => n.category === cat && n.is_parent === true);
        if (!parentNode) {
            parentNode = {
                id: `virtual_${cat}`,
                title: catTitles[cat] || cat.toUpperCase(),
                category: cat,
                is_virtual: true,
                x: width / 2,
                y: height / 2
            };
            finalNodes.push(parentNode);
        }

        if (coreNode && parentNode.id !== coreNode.id) {
            finalLinks.push({ source: coreNode.id, target: parentNode.id, is_main: true });
        }

        finalNodes.forEach(n => {
            if (n.category === cat && n.id !== parentNode.id && !n.is_virtual) {
                finalLinks.push({ source: parentNode.id, target: n.id, is_main: false });
            }
        });
    });

    svg = d3.select("#graphCanvas");
    svg.selectAll("*").remove();
    g = svg.append("g");

    const zoom = d3.zoom().on("zoom", (event) => {
        g.attr("transform", event.transform);
    });
    svg.call(zoom).on("dblclick.zoom", null);

    simulation = d3.forceSimulation(finalNodes)
        .force("link", d3.forceLink(finalLinks).id(d => d.id).distance(80))
        .force("charge", d3.forceManyBody().strength(-150))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(30));

    link = g.append("g").selectAll("line").data(finalLinks).join("line")
        .attr("stroke", "#444b5a").attr("stroke-opacity", 0.6);

    node = g.append("g").selectAll(".node").data(finalNodes).join("g")
        .attr("class", "node")
        .call(d3.drag()
            .on("start", (e, d) => {
                if (!e.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
            })
            .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on("end", (e, d) => {
                if (!e.active) simulation.alphaTarget(0);
                d.fx = null; d.fy = null;
            }));

    node.append("circle")
        .attr("r", d => d.category === 'core' ? 15 : 10)
        .attr("fill", d => colors[d.category] || colors['default'])
        .attr("stroke", "#16161a");

    node.append("text")
        .attr("dy", -15)
        .attr("text-anchor", "middle")
        .style("fill", "#94a1b2")
        .style("font-size", "12px")
        .text(d => d.title);

    node.on("click", (event, d) => {
        if (d.is_virtual) return;
        d3.selectAll("circle").attr("stroke", "#16161a").attr("stroke-width", 1);
        d3.select(event.currentTarget).select("circle").attr("stroke", "#7f5af0").attr("stroke-width", 3);

        // Вызов функции из интересов
        if (window.onNodeClick) window.onNodeClick(d.id);
    });

    simulation.on("tick", () => {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });
};

window.addNode = function() {
    editingNodeId = null;
    const modal = document.getElementById('nodeModal');
    if (modal) modal.style.display = 'flex';
};

window.onNodeClick = function(nodeId) {
    const list = document.getElementById('resultsList');
    if (list) list.innerHTML = '<div>Загрузка...</div>';

    // 1. Находим данные узла в симуляции D3, чтобы узнать его категорию
    const clickedNode = node.data().find(n => n.id === nodeId);
    const category = clickedNode ? clickedNode.category : 'psychology';

    console.log(`Клик по узлу ${nodeId}, категория: ${category}`);

    // 2. Добавляем категорию в запрос как Query Parameter
    fetch(`/api/graph/match/${nodeId}?category=${category}`)
        .then(res => res.json())
        .then(matches => {
            if (window.renderMatches) {
                window.renderMatches(matches);
            } else {
                // Если функции рендера нет в глобале, выведем просто список
                list.innerHTML = matches.map(m => `<div>${m.user_name} - ${m.compatibility}%</div>`).join('');
            }
        })
        .catch(err => {
            console.error("Ошибка поиска:", err);
            if (list) list.innerHTML = '<div>Ошибка загрузки</div>';
        });
};
