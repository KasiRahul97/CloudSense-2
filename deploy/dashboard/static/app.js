// CloudSense dashboard front-end.
// Talks to the local proxy (/api/*), which forwards to the model API.
let cpuValues = [];
let chartInstance = null;
let autoSimInterval = null;
let fleetSyncInterval = null;
let horizonLabel = "+4h";
let lookBack = 48;

let holdTicks = 0;
let scaleOutPrompted = false;
let extraInstanceId = null;

const ELEMENTS = {
    btnPredict: document.getElementById('btnPredictOnce'),
    toggleAuto: document.getElementById('toggleAutoSim'),
    thresholdInput: document.getElementById('thresholdInput'),
    manualCpuInput: document.getElementById('manualCpuInput'),

    valMAE: document.getElementById('valMAE'),
    valRMSE: document.getElementById('valRMSE'),
    valR2: document.getElementById('valR2'),
    valLookback: document.getElementById('valLookback'),
    healthStatus: document.getElementById('healthStatus'),
    modelBadge: document.getElementById('modelBadge'),

    currentCpuVal: document.getElementById('currentCpuVal'),
    predictedCpuVal: document.getElementById('predictedCpuVal'),
    recommendationVal: document.getElementById('recommendationVal'),
    recommendationCard: document.getElementById('recommendationCard'),

    fleetList: document.getElementById('fleetList'),
    fleetCountBadge: document.getElementById('fleetCountBadge'),

    console: document.getElementById('consoleWindow'),
    ctx: document.getElementById('cpuChart').getContext('2d'),

    modal: document.getElementById('approvalModal'),
    modalTitle: document.getElementById('modalTitle'),
    modalMsg: document.getElementById('modalMessage'),
    btnApprove: document.getElementById('btnApprove'),
    btnReject: document.getElementById('btnReject')
};

// Modals
let pendingAction = null;
function popModal(title, msg, actionCallback) {
    ELEMENTS.modalTitle.innerHTML = title;
    ELEMENTS.modalMsg.innerHTML = msg;
    pendingAction = actionCallback;
    ELEMENTS.modal.classList.add('active');
}
ELEMENTS.btnReject.onclick = () => {
    ELEMENTS.modal.classList.remove('active');
    log("SYSTEM", "User rejected scaling action.", "warn");
};
ELEMENTS.btnApprove.onclick = () => {
    ELEMENTS.modal.classList.remove('active');
    if (pendingAction) pendingAction();
};

async function init() {
    log("SYSTEM", "Initializing CloudSense dashboard...", "info");
    initChart();
    await fetchHealthAndMetrics();
    await seedFromSample();
    syncFleet();
    fleetSyncInterval = setInterval(syncFleet, 3000);

    ELEMENTS.btnPredict.addEventListener('click', triggerPrediction);
    ELEMENTS.toggleAuto.addEventListener('change', (e) => {
        if (e.target.checked) {
            log("SYSTEM", "Auto-simulator started", "info");
            triggerPrediction();
            autoSimInterval = setInterval(triggerPrediction, 4000);
        } else {
            log("SYSTEM", "Auto-simulator stopped", "warn");
            clearInterval(autoSimInterval);
        }
    });
}

// Seed the live window with REAL data from the cached NAB dataset.
async function seedFromSample() {
    try {
        const r = await fetch('/api/sample');
        const d = await r.json();
        cpuValues = d.cpu_percent.slice(-lookBack);
        log("DATA", `Loaded ${cpuValues.length}-step window (${d.source})`, "info");
    } catch (e) {
        cpuValues = Array.from({ length: lookBack }, () => 30 + Math.random() * 5);
        log("DATA", "Sample unavailable; using synthetic seed", "warn");
    }
    ELEMENTS.currentCpuVal.textContent = cpuValues[cpuValues.length - 1].toFixed(2) + "%";
    updateChart(cpuValues, null);
}

// Simulated autoscaling fleet (local demo, not real AWS)
async function syncFleet() {
    try {
        const r = await fetch('/api/fleet');
        renderFleet(await r.json());
    } catch (e) {}
}

function renderFleet(fleetDict) {
    ELEMENTS.fleetList.innerHTML = '';
    const keys = Object.keys(fleetDict);
    ELEMENTS.fleetCountBadge.textContent = `Instances: ${keys.length}`;
    keys.forEach(k => {
        const inst = fleetDict[k];
        let buttons = '';
        if (inst.status === 'Running') {
            buttons = `<button class="btn btn-secondary btn-small" onclick="fleetAction('${k}','stop')">Stop</button>`;
        } else if (inst.status === 'Stopped') {
            buttons = `<button class="btn btn-secondary btn-small" onclick="fleetAction('${k}','start')">Start</button>`;
        }
        let killBtn = '';
        if (!inst.is_main) {
            killBtn = `<button class="btn btn-danger btn-small" onclick="fleetAction('${k}','terminate')">Kill</button>`;
        }
        ELEMENTS.fleetList.insertAdjacentHTML('beforeend', `
            <div class="instance-row">
                <div class="instance-info">
                    <strong><span class="inst-status ${inst.status}"></span>${inst.name}</strong>
                    <span>[${inst.id}] - ${inst.type}</span>
                </div>
                <div class="instance-actions">${buttons}${killBtn}</div>
            </div>`);
    });
}

async function fleetAction(id, action) {
    log("FLEET", `${action.toUpperCase()} instance ${id} (simulated)`, "req");
    if (action === 'terminate') { extraInstanceId = null; scaleOutPrompted = false; holdTicks = 0; }
    await fetch(`/api/fleet/${id}/${action}`, { method: 'POST' });
    syncFleet();
}

async function provisionInstance() {
    log("FLEET", "Provisioning a simulated worker instance...", "req");
    const r = await fetch('/api/fleet/provision', { method: 'POST' });
    const data = await r.json();
    extraInstanceId = data.id;
    log("FLEET", `Created simulated instance ${data.id}`, "success");
    syncFleet();
}

async function fetchHealthAndMetrics() {
    try {
        const dataHealth = await (await fetch('/api/health')).json();
        if (!dataHealth.model_loaded) throw new Error(dataHealth.error || "model not loaded");
        ELEMENTS.healthStatus.textContent = "● API online";
        ELEMENTS.healthStatus.classList.remove('offline');
        ELEMENTS.modelBadge.textContent = "Model: " + (dataHealth.model || "CEEMDAN+CNN-BiLSTM");
        ELEMENTS.valLookback.textContent = dataHealth.look_back;
        lookBack = dataHealth.look_back || 48;
        if (dataHealth.horizon_label) horizonLabel = "+" + dataHealth.horizon_label;

        const dataMet = await (await fetch('/api/metrics')).json();
        const m = dataMet.metrics || {};
        ELEMENTS.valMAE.textContent = m.MAE != null ? parseFloat(m.MAE).toFixed(3) : "--";
        ELEMENTS.valRMSE.textContent = m.RMSE != null ? parseFloat(m.RMSE).toFixed(3) : "--";
        ELEMENTS.valR2.textContent = m.R2 != null ? parseFloat(m.R2).toFixed(4) : "--";
        if (chartInstance) {
            chartInstance.data.datasets[1].label = `Forecast (${horizonLabel})`;
            chartInstance.update();
        }
    } catch (e) {
        ELEMENTS.healthStatus.textContent = "● Offline";
        ELEMENTS.healthStatus.classList.add('offline');
        log("ERROR", "Inference API offline: " + e.message, "error");
    }
}

async function triggerPrediction() {
    let nextVal;
    const manualInputStr = ELEMENTS.manualCpuInput.value;
    if (manualInputStr !== "") {
        nextVal = parseFloat(manualInputStr);
        ELEMENTS.manualCpuInput.value = "";
    } else {
        const lastVal = cpuValues[cpuValues.length - 1];
        nextVal = lastVal + ((Math.random() - 0.5) * 12);
        if (Math.random() > 0.85) nextVal += 35;
        if (Math.random() > 0.85) nextVal -= 35;
    }
    nextVal = Math.max(0, Math.min(100, nextVal));

    cpuValues.push(nextVal);
    if (cpuValues.length > lookBack) cpuValues.shift();

    ELEMENTS.currentCpuVal.textContent = nextVal.toFixed(2) + "%";
    const threshold = parseFloat(ELEMENTS.thresholdInput.value) || 70.0;

    if (cpuValues.length < lookBack) {
        log("WARN", `Need ${lookBack} points; have ${cpuValues.length}`, "warn");
        return;
    }

    try {
        const res = await fetch('/api/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cpu_percent: cpuValues })
        });
        if (!res.ok) { log("ERROR", `Predict failed (${res.status})`, "error"); return; }
        const data = await res.json();

        const predCpu = parseFloat(data.predicted_cpu_percent);
        ELEMENTS.predictedCpuVal.textContent = predCpu.toFixed(2) + "%";
        log("MODEL", `Forecast ${horizonLabel}: ${predCpu.toFixed(2)}% | API rec: ${data.recommendation}`, "info");

        // Client-side scaling decision driven by the user's threshold.
        const scaleUp = predCpu >= threshold;
        ELEMENTS.recommendationVal.textContent = scaleUp ? "SCALE UP" : "HOLD";

        if (scaleUp) {
            ELEMENTS.recommendationCard.className = "stat-card glass-panel action-card scale_out";
            log("ALERT", `High load forecast ${horizonLabel} -> ${predCpu.toFixed(2)}%`, "error");
            holdTicks = 0;
            if (!scaleOutPrompted && !extraInstanceId) {
                scaleOutPrompted = true;
                popModal("⚠️ High Load Forecast",
                    `The model forecasts CPU reaching <b>${predCpu.toFixed(2)}%</b> in ${horizonLabel}.<br>Provision a (simulated) worker instance to absorb the load?`,
                    provisionInstance);
            }
        } else {
            ELEMENTS.recommendationCard.className = "stat-card glass-panel action-card hold";
            if (extraInstanceId) {
                if (predCpu < Math.max(threshold - 30, 10)) {
                    if (++holdTicks >= 3) {
                        holdTicks = 0;
                        popModal("✅ Load Normalized",
                            `Forecast CPU has stayed low (<b>${predCpu.toFixed(2)}%</b>).<br>Terminate the extra (simulated) instance to save cost?`,
                            () => fleetAction(extraInstanceId, 'terminate'));
                    }
                } else { holdTicks = 0; }
            }
        }
        updateChart(cpuValues, predCpu);
    } catch (e) {
        log("ERROR", "Prediction request failed", "error");
    }
}

function initChart() {
    const data = {
        labels: Array.from({ length: lookBack + 1 }, (_, i) => `T${i - lookBack}`),
        datasets: [
            { label: 'Input window CPU (%)', data: [], borderColor: 'rgba(255,255,255,0.4)', backgroundColor: 'rgba(255,255,255,0.05)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4 },
            { label: `Forecast (${horizonLabel})`, data: [], borderColor: '#4facfe', backgroundColor: '#4facfe', borderWidth: 3, pointRadius: 6, pointHoverRadius: 8, pointBackgroundColor: '#00d2ff', fill: false, borderDash: [5, 5] }
        ]
    };
    chartInstance = new Chart(ELEMENTS.ctx, {
        type: 'line', data: data,
        options: {
            responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
            plugins: { legend: { labels: { color: '#8b92a5' } } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b92a5', maxTicksLimit: 10 } },
                y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b92a5' } }
            }
        }
    });
}

function updateChart(hArr, pVal) {
    if (!chartInstance) return;
    const n = hArr.length;
    const hData = [...hArr, null];
    const pData = new Array(n + 1).fill(null);
    if (pVal !== null && pVal !== undefined) { pData[n - 1] = hArr[n - 1]; pData[n] = pVal; }
    chartInstance.data.labels = Array.from({ length: n + 1 }, (_, i) => i < n ? `T${i - n + 1}` : horizonLabel);
    chartInstance.data.datasets[0].data = hData;
    chartInstance.data.datasets[1].data = pData;
    chartInstance.update();
}

function log(tag, msg, type = "info") {
    const el = document.createElement('div');
    el.className = `log-line log-${type}`;
    el.innerHTML = `<span class="log-time">[${new Date().toTimeString().split(' ')[0]}]</span> <strong>[${tag}]</strong> ${msg}`;
    ELEMENTS.console.appendChild(el);
    ELEMENTS.console.scrollTop = ELEMENTS.console.scrollHeight;
}
function clearLogs() { ELEMENTS.console.innerHTML = ''; }
init();
