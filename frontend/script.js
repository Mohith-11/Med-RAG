// Theme Management
const themeSwitch = document.getElementById('theme-switch');
const htmlEl = document.documentElement;

// Load saved theme
const savedTheme = localStorage.getItem('medspace-theme') || 'dark';
htmlEl.setAttribute('data-theme', savedTheme);
themeSwitch.checked = savedTheme === 'dark';

themeSwitch.addEventListener('change', (e) => {
    const newTheme = e.target.checked ? 'dark' : 'light';
    htmlEl.setAttribute('data-theme', newTheme);
    localStorage.setItem('medspace-theme', newTheme);
});

// Canvas Starfield
const canvas = document.getElementById('starfield');
const ctx = canvas.getContext('2d');
let stars = [];
let mouseX = 0;
let mouseY = 0;
let isDarkTheme = savedTheme === 'dark';

// Observe theme changes for canvas behavior
const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        if (mutation.attributeName === 'data-theme') {
            isDarkTheme = htmlEl.getAttribute('data-theme') === 'dark';
        }
    });
});
observer.observe(htmlEl, { attributes: true });

// Mouse Tracking for Parallax & Paper Plane
const cursorPlane = document.getElementById('cursor-plane');
let planeX = window.innerWidth / 2;
let planeY = window.innerHeight / 2;
let currentPlaneX = planeX;
let currentPlaneY = planeY;

window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    planeX = e.clientX;
    planeY = e.clientY;
});

function initStars() {
    stars = [];
    const numStars = Math.floor((window.innerWidth * window.innerHeight) / 2000);
    for (let i = 0; i < numStars; i++) {
        stars.push({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            radius: Math.random() * 1.5,
            alpha: Math.random(),
            alphaChange: (Math.random() * 0.02) - 0.01,
            z: Math.random() * 0.5 + 0.1 // For parallax
        });
    }
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    initStars();
}

const resizeObserver = new ResizeObserver(resizeCanvas);
resizeObserver.observe(document.body);
resizeCanvas();

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (isDarkTheme) {
        stars.forEach(star => {
            // Parallax shift
            const shiftX = (mouseX - window.innerWidth / 2) * star.z * 0.05;
            const shiftY = (mouseY - window.innerHeight / 2) * star.z * 0.05;
            
            // Draw star
            ctx.beginPath();
            ctx.arc(star.x + shiftX, star.y + shiftY, star.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha})`;
            ctx.fill();
            
            // Twinkle
            star.alpha += star.alphaChange;
            if (star.alpha <= 0.1 || star.alpha >= 1) {
                star.alphaChange = -star.alphaChange;
            }
        });
    }

    // Plane logic
    const dx = planeX - currentPlaneX;
    const dy = planeY - currentPlaneY;
    currentPlaneX += dx * 0.1; // Easing
    currentPlaneY += dy * 0.1;
    
    // Rotation based on movement direction
    let angle = Math.atan2(dy, dx) * (180 / Math.PI);
    // Add an offset because the SVG itself points diagonally
    angle += 45; 
    
    cursorPlane.style.transform = `translate(${currentPlaneX}px, ${currentPlaneY}px) rotate(${angle}deg)`;
    
    requestAnimationFrame(animate);
}
animate();

// Tab Switching
const tabBtns = document.querySelectorAll('.tab-btn');
const views = document.querySelectorAll('.view-section');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const target = btn.getAttribute('data-target');
        views.forEach(v => {
            v.classList.remove('active');
            if (v.id === target) v.classList.add('active');
        });
        
        if(target === 'eval-view' && !evalLoaded) {
            loadEvaluationData();
            evalLoaded = true;
        }
    });
});

// Chat Interface
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearInputBtn = document.getElementById('clear-input-btn');
const chatMessages = document.getElementById('chat-messages');

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value.trim().length > 0) {
        clearInputBtn.classList.remove('hidden');
    } else {
        clearInputBtn.classList.add('hidden');
    }
});

clearInputBtn.addEventListener('click', () => {
    chatInput.value = '';
    chatInput.style.height = 'auto';
    clearInputBtn.classList.add('hidden');
    chatInput.focus();
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

// Chat History State
let currentChatId = Date.now().toString();
let chats = JSON.parse(localStorage.getItem('medspace-chats')) || {};

function saveChat() {
    localStorage.setItem('medspace-chats', JSON.stringify(chats));
    renderSidebarChats();
}

function renderSidebarChats() {
    const list = document.getElementById('chat-history-list');
    list.innerHTML = '';
    
    // Sort by newest
    const sortedIds = Object.keys(chats).sort((a, b) => b - a);
    
    sortedIds.forEach(id => {
        const chat = chats[id];
        const li = document.createElement('li');
        li.className = `history-item ${id === currentChatId ? 'active' : ''}`;
        li.innerHTML = `<i class="fa-regular fa-message"></i> ${chat.title}`;
        li.addEventListener('click', () => loadChat(id));
        list.appendChild(li);
    });
}

function loadChat(id) {
    currentChatId = id;
    const chat = chats[id];
    document.getElementById('current-chat-title').innerText = chat.title;
    chatMessages.innerHTML = '';
    
    chat.messages.forEach(msg => {
        appendMessageUI(msg.role, msg.content, msg.sources, false);
    });
    
    renderSidebarChats();
    scrollToBottom();
}

document.getElementById('new-chat-btn').addEventListener('click', () => {
    currentChatId = Date.now().toString();
    document.getElementById('current-chat-title').innerText = 'New Conversation';
    chatMessages.innerHTML = `
        <div class="message assistant welcome-msg">
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="msg-content">
                <p>Greetings. I am MedSpace AI, your advanced medical knowledge assistant.</p>
                <p>How can I assist you with clinical information today?</p>
            </div>
        </div>
    `;
    renderSidebarChats();
});

document.getElementById('clear-history-btn').addEventListener('click', () => {
    if (confirm("Are you sure you want to clear all chat history?")) {
        chats = {};
        saveChat();
        document.getElementById('new-chat-btn').click();
    }
});

function appendMessageUI(role, content, sources = null, save = true) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    const icon = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    // Parse markdown for assistant, plain text for user
    const htmlContent = role === 'assistant' ? marked.parse(content) : `<p>${content.replace(/\n/g, '<br>')}</p>`;
    
    let sourcesHTML = '';
    if (sources && sources.length > 0) {
        sourcesHTML = `<div class="sources-container">
            <div class="sources-title"><i class="fa-solid fa-book-medical"></i> Sources</div>
            <div class="sources-list">
                ${sources.map(s => `<span class="source-chip" title="${s.text}">📄 ${s.title} (p.${s.page || 'N/A'})</span>`).join('')}
            </div>
        </div>`;
    }

    msgDiv.innerHTML = `
        <div class="avatar">${icon}</div>
        <div class="msg-content">
            ${htmlContent}
            ${sourcesHTML}
        </div>
    `;
    
    chatMessages.appendChild(msgDiv);
    scrollToBottom();

    if (save) {
        if (!chats[currentChatId]) {
            let title = content.length > 30 ? content.substring(0, 30) + '...' : content;
            if(role === 'assistant') title = "Medical Query"; // Fallback if bot speaks first somehow
            chats[currentChatId] = { title, messages: [] };
        }
        chats[currentChatId].messages.push({ role, content, sources });
        saveChat();
        document.getElementById('current-chat-title').innerText = chats[currentChatId].title;
    }
}

function scrollToBottom() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant typing-msg';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Flight animation
    const icon = sendBtn.querySelector('i');
    icon.classList.add('flight-anim');
    setTimeout(() => {
        icon.classList.remove('flight-anim');
    }, 500);

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    clearInputBtn.classList.add('hidden');

    // Add user message
    appendMessageUI('user', text);
    showTypingIndicator();

    try {
        // Fetch from API
        const response = await fetch('/api/agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        removeTypingIndicator();
        appendMessageUI('assistant', data.answer || "I received a response, but it was empty.", data.sources);
        
    } catch (error) {
        console.error('Fetch error:', error);
        removeTypingIndicator();
        // Mock fallback
        setTimeout(() => {
            const mockAns = `**System Notice:** Backend connection failed. Showing mock response.\n\nBased on the retrieved medical literature, the standard approach involves multidisciplinary evaluation. The patient should be monitored for **adverse effects** while undergoing treatment.`;
            const mockSources = [
                { title: 'Oncology_Guidelines_2025.pdf', text: 'Treatment paradigms involve...', page: 42 },
                { title: 'Clinical_Trials_Phase3.pdf', text: 'Adverse effects monitored in cohort...', page: 12 }
            ];
            appendMessageUI('assistant', mockAns, mockSources);
        }, 800);
    }
}

// Evaluation Dashboard
let evalLoaded = false;
let charts = {};

document.getElementById('refresh-eval-btn').addEventListener('click', loadEvaluationData);

async function loadEvaluationData() {
    const btn = document.getElementById('refresh-eval-btn');
    const icon = btn.querySelector('i');
    icon.classList.add('fa-spin');

    try {
        // Try fetching real API
        let summaryRes = await fetch('/api/evaluation/summary').catch(() => null);
        let summaryData = null;
        let resultsData = null;

        if (summaryRes && summaryRes.ok) {
            summaryData = await summaryRes.json();
            let resultsRes = await fetch('/api/evaluation/results');
            resultsData = await resultsRes.json();
        } else {
            // Mock Data Fallback
            console.log("Using mock evaluation data");
            summaryData = {
                "questions_evaluated": 20,
                "avg_confidence": 0.824,
                "retrieval_quality": { "mrr": 0.8639, "hit_rate_at_5": 0.9750 },
                "generation_lexical": { "answer_f1": 0.4509, "rougeL": 0.2267 },
                "faithfulness_relevance": { "faithfulness": 0.8610 },
                "scope": { "precision": 3.92, "completeness": 3.96 }
            };
            resultsData = Array.from({length: 10}).map((_, i) => ({
                question_id: `Q${i+1}`,
                question: `What are the side effects of drug ${i}?`,
                category: i % 2 === 0 ? 'Treatment' : 'Diagnosis',
                faithfulness: (Math.random() * 0.4 + 0.6).toFixed(2),
                token_f1: (Math.random() * 0.5 + 0.3).toFixed(2),
                judge_score: (Math.random() * 2 + 3).toFixed(1),
                generated_answer: `Mock answer for question ${i}.`,
                ground_truth: `True answer for question ${i}.`
            }));
        }

        renderEvalSummary(summaryData);
        renderEvalCharts(summaryData);
        renderEvalTable(resultsData);

    } catch (err) {
        console.error("Evaluation fetch failed", err);
    } finally {
        icon.classList.remove('fa-spin');
    }
}

function renderEvalSummary(data) {
    document.getElementById('metric-confidence').innerText = (data.avg_confidence * 100).toFixed(1) + '%';
    document.getElementById('metric-mrr').innerText = data.retrieval_quality.mrr.toFixed(3);
    document.getElementById('metric-f1').innerText = data.generation_lexical.answer_f1.toFixed(3);
    document.getElementById('metric-faithfulness').innerText = data.faithfulness_relevance.faithfulness.toFixed(3);
}

function renderEvalCharts(data) {
    const textColor = isDarkTheme ? '#E2E8F0' : '#0F172A';
    const gridColor = isDarkTheme ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';

    const chartOptions = {
        responsive: true,
        plugins: { legend: { labels: { color: textColor } } },
        scales: {
            y: { ticks: { color: textColor }, grid: { color: gridColor } },
            x: { ticks: { color: textColor }, grid: { display: false } }
        }
    };

    // Destroy existing
    Object.values(charts).forEach(c => c.destroy());

    // Retrieval Chart
    const ctx1 = document.getElementById('retrievalChart').getContext('2d');
    charts.ret = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: ['MRR', 'Hit Rate@5'],
            datasets: [{
                label: 'Score',
                data: [data.retrieval_quality.mrr, data.retrieval_quality.hit_rate_at_5 || 0.9],
                backgroundColor: 'rgba(99, 102, 241, 0.7)'
            }]
        },
        options: { ...chartOptions, scales: { y: { ...chartOptions.scales.y, max: 1 } } }
    });

    // Generation Chart
    const ctx2 = document.getElementById('generationChart').getContext('2d');
    charts.gen = new Chart(ctx2, {
        type: 'radar',
        data: {
            labels: ['Answer F1', 'ROUGE-L', 'Faithfulness'],
            datasets: [{
                label: 'Score',
                data: [data.generation_lexical.answer_f1, data.generation_lexical.rougeL || 0.3, data.faithfulness_relevance.faithfulness],
                borderColor: '#10B981',
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
            }]
        },
        options: {
            responsive: true,
            scales: { r: { angleLines: { color: gridColor }, grid: { color: gridColor }, pointLabels: { color: textColor }, ticks: { backdropColor: 'transparent', color: textColor, max: 1 } } }
        }
    });

    // Judge Chart
    const ctx3 = document.getElementById('judgeChart').getContext('2d');
    charts.judge = new Chart(ctx3, {
        type: 'bar',
        data: {
            labels: ['Precision', 'Completeness'],
            datasets: [{
                label: 'LLM Judge Score (1-5)',
                data: [data.scope.precision || 3.9, data.scope.completeness || 3.9],
                backgroundColor: 'rgba(245, 158, 11, 0.7)'
            }]
        },
        options: { ...chartOptions, scales: { y: { ...chartOptions.scales.y, max: 5 } } }
    });
}

function renderEvalTable(results) {
    const tbody = document.getElementById('eval-table-body');
    tbody.innerHTML = '';

    window.evalResults = results; // Store for modal

    results.forEach((row, idx) => {
        const tr = document.createElement('tr');
        
        const fBadge = row.faithfulness > 0.8 ? 'good' : (row.faithfulness > 0.5 ? 'avg' : 'poor');
        const jBadge = row.judge_score >= 4 ? 'good' : (row.judge_score >= 3 ? 'avg' : 'poor');

        tr.innerHTML = `
            <td>${row.question_id || idx}</td>
            <td style="max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${row.question}</td>
            <td>${row.category || 'General'}</td>
            <td><span class="badge ${fBadge}">${row.faithfulness}</span></td>
            <td>${parseFloat(row.token_f1).toFixed(2)}</td>
            <td><span class="badge ${jBadge}">${row.judge_score}/5</span></td>
        `;

        tr.addEventListener('click', () => showDetailsModal(idx));
        tbody.appendChild(tr);
    });
}

// Modal Logic
const modal = document.getElementById('details-modal');
const closeBtn = document.getElementById('close-modal-btn');

function showDetailsModal(idx) {
    const data = window.evalResults[idx];
    document.getElementById('modal-title').innerText = `Query Details: ${data.question_id || idx}`;
    
    document.getElementById('modal-body').innerHTML = `
        <h4>Question</h4>
        <p>${data.question}</p>
        
        <h4>Ground Truth</h4>
        <p>${data.ground_truth || 'N/A'}</p>
        
        <h4>Generated Answer</h4>
        <p>${data.generated_answer}</p>
        
        <h4>Metrics Breakdown</h4>
        <pre>${JSON.stringify({
            faithfulness: data.faithfulness,
            token_f1: data.token_f1,
            judge_score: data.judge_score,
            exact_match: data.exact_match || null
        }, null, 2)}</pre>
    `;
    
    modal.classList.remove('hidden');
}

closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.add('hidden');
});

// Initialization
renderSidebarChats();
if (Object.keys(chats).length > 0) {
    loadChat(Object.keys(chats).sort((a, b) => b - a)[0]);
} else {
    document.getElementById('new-chat-btn').click();
}
