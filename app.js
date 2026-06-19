/* ==========================================================
   Roulette Predictor — clean rewrite
   Focus: Color  +  Even/Odd prediction
   Strategy: backtest-weighted ensemble (Markov + n-gram + EWMA)
   ========================================================== */

(() => {

// European wheel red numbers
const RED = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

// ---------- helpers ----------
const colorOf  = n => n === 0 ? 'green' : (RED.has(n) ? 'red' : 'black');
const parityOf = n => n === 0 ? 'zero' : (n % 2 === 0 ? 'even' : 'odd');

// label kind → list of classes used
const COLOR_CLASSES  = ['red','black','green'];
const PARITY_CLASSES = ['even','odd','zero'];

// ---------- state ----------
const State = {
    rounds: [],         // [{number, color, parity, time}]
    predictions: [],    // [{predColor, actColor, predParity, actParity, t}]
    lastPrediction: null,
    lastInsight: null,
    tab: 'predict',
    mode: 'tap'
};

// ---------- storage ----------
const KEY_R = 'rp_rounds_v2';
const KEY_P = 'rp_preds_v2';

function save() {
    try {
        localStorage.setItem(KEY_R, JSON.stringify(State.rounds));
        localStorage.setItem(KEY_P, JSON.stringify(State.predictions));
    } catch(_) {}
}
function load() {
    try {
        const r = JSON.parse(localStorage.getItem(KEY_R) || '[]');
        const p = JSON.parse(localStorage.getItem(KEY_P) || '[]');
        if (Array.isArray(r)) State.rounds = r;
        if (Array.isArray(p)) State.predictions = p;
    } catch(_) {}
}

// ==========================================================
// PREDICTION ENGINE
// ----------------------------------------------------------
// Three sub-models, each predicts a probability distribution
// over outcome classes. They are blended using weights
// derived from a walk-forward backtest on the user data:
// each model gets a weight proportional to how often it
// would have been correct on the training history.
// ----------------------------------------------------------
// Sub-models:
//  M1  Laplace-smoothed 1st-order Markov chain
//  M2  N-gram pattern lookup (looks back up to N=6 events,
//      finds matching sub-sequence in history, sees what
//      followed; uses longest match available, weighted by
//      length)
//  M3  Exponentially weighted recent frequency (EWMA)
// ----------------------------------------------------------
// Output: { winner, probabilities, confidence, insight }
// ==========================================================

const NGRAM_MAX = 6;
const EWMA_HALFLIFE = 25; // rounds
const LAPLACE_ALPHA = 1;

function uniform(classes) {
    const p = {};
    classes.forEach(c => p[c] = 1 / classes.length);
    return p;
}

function normalize(scores, classes) {
    let s = 0;
    classes.forEach(c => s += (scores[c] || 0));
    const out = {};
    if (s <= 0) return uniform(classes);
    classes.forEach(c => out[c] = (scores[c] || 0) / s);
    return out;
}

// ---- M1: 1st-order Markov ----
function markovProb(seq, classes) {
    if (seq.length < 2) return uniform(classes);
    const trans = {};
    classes.forEach(a => {
        trans[a] = {};
        classes.forEach(b => trans[a][b] = LAPLACE_ALPHA);
    });
    for (let i = 1; i < seq.length; i++) trans[seq[i-1]][seq[i]]++;
    const last = seq[seq.length - 1];
    return normalize({ ...trans[last] }, classes);
}

// ---- M2: variable-length n-gram lookup ----
function ngramProb(seq, classes) {
    if (seq.length < 3) return uniform(classes);
    const scores = {};
    classes.forEach(c => scores[c] = LAPLACE_ALPHA);

    // Try patterns from longest to shortest
    for (let len = Math.min(NGRAM_MAX, seq.length - 1); len >= 2; len--) {
        const pattern = seq.slice(-len);
        let matches = 0;
        // Find pattern in earlier sequence (excluding the final element)
        for (let i = 0; i <= seq.length - len - 1; i++) {
            let ok = true;
            for (let j = 0; j < len; j++) {
                if (seq[i + j] !== pattern[j]) { ok = false; break; }
            }
            if (ok) {
                const next = seq[i + len];
                // Longer matches weighted more
                scores[next] = (scores[next] || 0) + Math.pow(len, 1.5);
                matches++;
            }
        }
        // If we found enough matches at this length, stop scanning shorter
        if (matches >= 3) break;
    }
    return normalize(scores, classes);
}

// ---- M3: EWMA recent frequency ----
function ewmaProb(seq, classes) {
    if (seq.length === 0) return uniform(classes);
    const counts = {};
    classes.forEach(c => counts[c] = LAPLACE_ALPHA);
    const n = seq.length;
    const lambda = Math.log(2) / EWMA_HALFLIFE;
    for (let i = 0; i < n; i++) {
        const age = n - 1 - i; // 0 = most recent
        const w = Math.exp(-lambda * age);
        counts[seq[i]] = (counts[seq[i]] || 0) + w;
    }
    return normalize(counts, classes);
}

// ---- ensemble weights via walk-forward backtest ----
// For each i in [warmup..n-1], use seq[0..i-1] to predict seq[i].
// Count how often each model was correct. Weights ∝ accuracy.
function backtestWeights(seq, classes, warmup = 8) {
    const models = ['m1','m2','m3'];
    const hits = { m1: 0, m2: 0, m3: 0 };
    let trials = 0;
    if (seq.length <= warmup + 1) {
        return { m1: 1, m2: 1, m3: 1, trials: 0, hits };
    }
    for (let i = warmup; i < seq.length; i++) {
        const past = seq.slice(0, i);
        const actual = seq[i];
        const p1 = markovProb(past, classes);
        const p2 = ngramProb(past, classes);
        const p3 = ewmaProb(past, classes);
        const argmax = (p) => classes.reduce((a, b) => p[a] >= p[b] ? a : b);
        if (argmax(p1) === actual) hits.m1++;
        if (argmax(p2) === actual) hits.m2++;
        if (argmax(p3) === actual) hits.m3++;
        trials++;
    }
    // Smoothed weights: add 1 hit to each so a model with 0
    // observed wins still has a nonzero floor
    return {
        m1: (hits.m1 + 1) / (trials + 3),
        m2: (hits.m2 + 1) / (trials + 3),
        m3: (hits.m3 + 1) / (trials + 3),
        trials,
        hits
    };
}

// ---- main predict for one outcome kind ----
function predictKind(seq, classes) {
    if (seq.length === 0) {
        return { winner: classes[0], probs: uniform(classes), confidence: 0, weights: null, hits: null };
    }
    const p1 = markovProb(seq, classes);
    const p2 = ngramProb(seq, classes);
    const p3 = ewmaProb(seq, classes);
    const w = backtestWeights(seq, classes);
    const wsum = (w.m1 + w.m2 + w.m3) || 1;

    const blended = {};
    classes.forEach(c => {
        blended[c] = (p1[c] * w.m1 + p2[c] * w.m2 + p3[c] * w.m3) / wsum;
    });

    // winner & confidence (margin-based)
    const sorted = classes.slice().sort((a, b) => blended[b] - blended[a]);
    const winner = sorted[0];
    const margin = blended[sorted[0]] - blended[sorted[1] || sorted[0]];
    // confidence: blends raw top-prob and margin; cap at 92
    const rawConf = blended[winner] * 100;
    const confidence = Math.min(92, Math.max(34, Math.round(rawConf * 0.6 + margin * 200)));

    return { winner, probs: blended, confidence, weights: w, hits: w.hits, trials: w.trials };
}

function runPrediction() {
    const r = State.rounds;
    if (r.length < 3) {
        toast('Add at least 3 rounds first', 'error');
        return null;
    }

    // Build sequences. Use 'red'/'black'/'green' for color and
    // 'even'/'odd'/'zero' for parity. The 'zero'/'green' classes
    // make the model honest about the green pocket.
    const colorSeq  = r.map(x => x.color);
    const paritySeq = r.map(x => x.parity);

    const colorRes  = predictKind(colorSeq,  COLOR_CLASSES);
    const parityRes = predictKind(paritySeq, PARITY_CLASSES);

    // Skip 'green' / 'zero' as final answer unless it overwhelmingly dominates,
    // since users want a usable bet
    const adjust = (res, classes, skip) => {
        if (res.winner === skip && res.probs[skip] < 0.45) {
            // pick best among non-skip
            const alt = classes.filter(c => c !== skip)
                .reduce((a, b) => res.probs[a] >= res.probs[b] ? a : b);
            res.winner = alt;
            res.confidence = Math.max(34, Math.round(res.probs[alt] * 100));
        }
        return res;
    };
    adjust(colorRes,  COLOR_CLASSES,  'green');
    adjust(parityRes, PARITY_CLASSES, 'zero');

    const pred = {
        color: colorRes.winner,
        colorConfidence: colorRes.confidence,
        colorProbs: colorRes.probs,
        parity: parityRes.winner,
        parityConfidence: parityRes.confidence,
        parityProbs: parityRes.probs,
        insight: {
            colorWeights: colorRes.weights,
            colorHits: colorRes.hits,
            parityWeights: parityRes.weights,
            parityHits: parityRes.hits,
            trials: colorRes.trials
        }
    };
    return pred;
}

// ==========================================================
//  UI RENDERING
// ==========================================================

const $ = id => document.getElementById(id);

function toast(msg, type = '') {
    const t = $('toast');
    t.textContent = msg;
    t.className = `toast show ${type}`;
    clearTimeout(t._timer);
    t._timer = setTimeout(() => t.classList.remove('show'), 2200);
}

function renderBadge() {
    $('roundsBadge').textContent = State.rounds.length;
}

function renderPrediction(p) {
    const cv = $('colorValue'), pv = $('parityValue');
    const cb = $('colorConfBar'), pb = $('parityConfBar');
    const ct = $('colorConfText'), pt = $('parityConfText');

    if (!p) {
        cv.className = 'pc-value'; cv.textContent = '—';
        pv.className = 'pc-value'; pv.textContent = '—';
        cb.style.width = '0%';      pb.style.width = '0%';
        ct.textContent = '—';        pt.textContent = '—';
        return;
    }

    const colorLabel = p.color.charAt(0).toUpperCase() + p.color.slice(1);
    cv.className = `pc-value ${p.color}`;
    cv.textContent = colorLabel;
    cb.style.width = `${p.colorConfidence}%`;
    ct.textContent = `${p.colorConfidence}%`;

    const parityLabel = p.parity === 'zero' ? 'Zero' : (p.parity === 'even' ? 'Even' : 'Odd');
    pv.className = `pc-value ${p.parity}`;
    pv.textContent = parityLabel;
    pb.style.width = `${p.parityConfidence}%`;
    pt.textContent = `${p.parityConfidence}%`;
}

function renderPredictNote() {
    const n = State.rounds.length;
    const el = $('predictNote');
    if (n < 3) {
        el.className = 'predict-note warn';
        el.textContent = `Need ${3 - n} more round${3 - n === 1 ? '' : 's'} before predicting.`;
    } else if (n < 30) {
        el.className = 'predict-note warn';
        el.textContent = `Model works best with ≥ 30 rounds. Currently ${n}. Upload training data for stronger accuracy.`;
    } else {
        el.className = 'predict-note';
        el.textContent = `Model trained on ${n} rounds.`;
    }
}

function renderAccuracyStrip() {
    const preds = State.predictions;
    if (preds.length === 0) {
        $('liveColorAcc').textContent  = '—';
        $('liveParityAcc').textContent = '—';
        $('livePredCount').textContent = '0';
        return;
    }
    const ca = preds.filter(p => p.predColor  === p.actColor).length;
    const pa = preds.filter(p => p.predParity === p.actParity).length;
    $('liveColorAcc').textContent  = `${Math.round(ca / preds.length * 100)}%`;
    $('liveParityAcc').textContent = `${Math.round(pa / preds.length * 100)}%`;
    $('livePredCount').textContent = preds.length;
}

function renderHistory() {
    const list = $('historyList');
    if (State.rounds.length === 0) {
        list.innerHTML = '<p class="empty-state">No rounds yet. Add some data to begin.</p>';
        return;
    }
    // Render at most 200 most-recent for performance
    const max = 200;
    const len = State.rounds.length;
    const start = Math.max(0, len - max);
    const items = [];
    for (let i = len - 1; i >= start; i--) {
        const r = State.rounds[i];
        const time = r.time ? new Date(r.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
        items.push(`
            <div class="history-item">
                <div class="history-chip ${r.color}">${r.number}</div>
                <div class="history-info">
                    <div class="history-num">#${i + 1} · ${r.color} · ${r.parity}</div>
                    <div class="history-meta">${time || 'No timestamp'}</div>
                </div>
                <button class="history-delete" data-idx="${i}" aria-label="Delete">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
            </div>
        `);
    }
    list.innerHTML = items.join('');
    list.querySelectorAll('.history-delete').forEach(b => {
        b.addEventListener('click', () => deleteRound(parseInt(b.dataset.idx, 10)));
    });
}

function renderStats() {
    const r = State.rounds;
    const n = r.length;
    const counts = { red: 0, black: 0, green: 0, even: 0, odd: 0, zero: 0 };
    for (const x of r) { counts[x.color]++; counts[x.parity]++; }

    const pct = c => n ? Math.round(counts[c] / n * 100) : 0;

    const rows = [
        ['red',   'Red'],
        ['black', 'Black'],
        ['green', 'Green'],
        ['even',  'Even'],
        ['odd',   'Odd']
    ];
    $('statRows').innerHTML = rows.map(([key, label]) => `
        <div class="stat-row">
            <div class="stat-key">${label}</div>
            <div class="stat-track"><div class="stat-fill ${key}" style="width:${pct(key)}%"></div></div>
            <div class="stat-value">${counts[key]}<span class="pct">${pct(key)}%</span></div>
        </div>
    `).join('');

    // Heatmap
    const freq = {};
    for (const x of r) freq[x.number] = (freq[x.number] || 0) + 1;
    const maxF = Math.max(1, ...Object.values(freq));
    const cells = [];
    // 0 first as wide row
    {
        const c = freq[0] || 0;
        const intensity = c / maxF;
        const bg = c === 0 ? '' : `background: rgba(22,163,74, ${0.25 + intensity * 0.7});`;
        cells.push(`<div class="heat-cell zero ${c > 0 ? 'hit' : ''}" style="${bg}" title="${c} hits">0 · ${c}</div>`);
    }
    for (let i = 1; i <= 36; i++) {
        const c = freq[i] || 0;
        const intensity = c / maxF;
        const col = colorOf(i);
        let bg = '';
        if (c > 0) {
            bg = col === 'red'
                ? `background: rgba(226,59,59, ${0.25 + intensity * 0.7});`
                : `background: rgba(31,41,55, ${0.25 + intensity * 0.7});`;
        }
        cells.push(`<div class="heat-cell ${c > 0 ? 'hit' : ''}" style="${bg}" title="${c} hits">${i}</div>`);
    }
    $('heatmap').innerHTML = cells.join('');

    // Insight
    const ins = State.lastInsight;
    const insightCard = $('insightCard');
    if (!ins) {
        insightCard.innerHTML = '<p class="empty-state">Predict at least once to see model insight.</p>';
    } else {
        const cw = ins.colorWeights;
        const pw = ins.parityWeights;
        const ch = ins.colorHits;
        const ph = ins.parityHits;
        const t = ins.trials;
        const totC = (cw.m1 + cw.m2 + cw.m3) || 1;
        const totP = (pw.m1 + pw.m2 + pw.m3) || 1;
        insightCard.innerHTML = `
            <div class="ins-row"><span class="ins-key">Backtest rounds</span><span class="ins-val">${t}</span></div>
            <div class="ins-row"><span class="ins-key">Markov weight (color · parity)</span><span class="ins-val">${(cw.m1/totC*100).toFixed(0)}% · ${(pw.m1/totP*100).toFixed(0)}%</span></div>
            <div class="ins-row"><span class="ins-key">N-gram weight (color · parity)</span><span class="ins-val">${(cw.m2/totC*100).toFixed(0)}% · ${(pw.m2/totP*100).toFixed(0)}%</span></div>
            <div class="ins-row"><span class="ins-key">EWMA weight (color · parity)</span><span class="ins-val">${(cw.m3/totC*100).toFixed(0)}% · ${(pw.m3/totP*100).toFixed(0)}%</span></div>
            <div class="ins-row"><span class="ins-key">Best color sub-model hits</span><span class="ins-val">${Math.max(ch.m1, ch.m2, ch.m3)} / ${t}</span></div>
            <div class="ins-row"><span class="ins-key">Best parity sub-model hits</span><span class="ins-val">${Math.max(ph.m1, ph.m2, ph.m3)} / ${t}</span></div>
        `;
    }
}

// Lazy renderers — only run for the visible tab
function renderActiveTab() {
    if (State.tab === 'history') renderHistory();
    else if (State.tab === 'stats') renderStats();
}

let rafScheduled = false;
function renderAll() {
    if (rafScheduled) return;
    rafScheduled = true;
    requestAnimationFrame(() => {
        rafScheduled = false;
        renderBadge();
        renderPredictNote();
        renderAccuracyStrip();
        renderActiveTab();
    });
}

// ==========================================================
//  ACTIONS
// ==========================================================

function addRound(num, customColor = null, customTime = null) {
    num = parseInt(num, 10);
    if (isNaN(num) || num < 0 || num > 36) {
        toast('Number must be 0–36', 'error');
        return false;
    }
    // Score any pending prediction
    if (State.lastPrediction) {
        const ac = customColor || colorOf(num);
        const ap = parityOf(num);
        State.predictions.push({
            predColor: State.lastPrediction.color,
            actColor: ac,
            predParity: State.lastPrediction.parity,
            actParity: ap,
            t: Date.now()
        });
        State.lastPrediction = null;
    }
    State.rounds.push({
        number: num,
        color: customColor || colorOf(num),
        parity: parityOf(num),
        time: customTime || Date.now()
    });
    save();

    // Last-added strip
    const la = $('lastAdded');
    if (la) {
        const last = State.rounds[State.rounds.length - 1];
        la.classList.add('show');
        la.innerHTML = `<div class="last-chip ${last.color}">${last.number}</div><span>Added · ${last.color} · ${last.parity}</span>`;
    }

    renderAll();
    return true;
}

function deleteRound(idx) {
    State.rounds.splice(idx, 1);
    save();
    renderAll();
}

function clearAll() {
    State.rounds = [];
    State.predictions = [];
    State.lastPrediction = null;
    State.lastInsight = null;
    save();
    $('lastAdded')?.classList.remove('show');
    renderPrediction(null);
    renderAll();
    toast('Cleared');
}

function exportJSON() {
    const data = {
        exportedAt: new Date().toISOString(),
        totalRounds: State.rounds.length,
        rounds: State.rounds
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `roulette-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Exported', 'success');
}

function importJSON(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const raw = JSON.parse(e.target.result);
            let arr = [];
            if (Array.isArray(raw)) arr = raw;
            else if (Array.isArray(raw.rounds)) arr = raw.rounds;
            else if (Array.isArray(raw.data)) arr = raw.data;
            else throw new Error('Unrecognized JSON shape');

            let added = 0;
            const baseTime = Date.now() - arr.length * 1000;
            for (let i = 0; i < arr.length; i++) {
                const it = arr[i];
                let num, customColor = null, customTime = null;
                if (typeof it === 'number') num = it;
                else if (it && typeof it === 'object') {
                    num = it.number !== undefined ? it.number : it.num;
                    if (typeof it.color === 'string') customColor = it.color.toLowerCase();
                    if (typeof it.time === 'number') customTime = it.time;
                    else if (typeof it.time === 'string') {
                        const t = Date.parse(it.time);
                        if (!isNaN(t)) customTime = t;
                    }
                }
                if (typeof num !== 'number' || num < 0 || num > 36 || isNaN(num)) continue;
                State.rounds.push({
                    number: parseInt(num, 10),
                    color: customColor || colorOf(parseInt(num, 10)),
                    parity: parityOf(parseInt(num, 10)),
                    time: customTime || (baseTime + i * 1000)
                });
                added++;
            }
            save();
            renderAll();
            toast(`Imported ${added} rounds`, 'success');
        } catch (err) {
            console.error(err);
            toast('Could not parse JSON', 'error');
        }
    };
    reader.readAsText(file);
}

function downloadSample() {
    const sample = {
        info: 'Roulette training data sample',
        rounds: [17, 32, 0, 21, 8, 14, 5, 19, 2, 26, 11, 33, 9, 22, 4, 30, 1, 13, 24, 35, 7, 16, 28, 3, 25, 12, 6, 18, 27, 10]
    };
    const blob = new Blob([JSON.stringify(sample, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'roulette-sample.json';
    a.click();
    URL.revokeObjectURL(a.href);
    toast('Sample saved', 'success');
}

// ==========================================================
//  WIRING
// ==========================================================

function buildNumberGrid() {
    const grid = $('numberGrid');
    let html = `<button class="num-chip green" data-num="0">0</button>`;
    for (let i = 1; i <= 36; i++) {
        html += `<button class="num-chip ${colorOf(i)}" data-num="${i}">${i}</button>`;
    }
    grid.innerHTML = html;
    grid.addEventListener('click', e => {
        const btn = e.target.closest('.num-chip');
        if (!btn) return;
        addRound(parseInt(btn.dataset.num, 10));
    });
}

function setupTabs() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = btn.dataset.tab;
            if (State.tab === t) return;
            State.tab = t;
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b === btn));
            document.querySelectorAll('.tab').forEach(s => s.classList.toggle('active', s.id === `tab-${t}`));
            renderActiveTab();
        });
    });
}

function setupModes() {
    document.querySelectorAll('.seg').forEach(btn => {
        btn.addEventListener('click', () => {
            const m = btn.dataset.mode;
            if (State.mode === m) return;
            State.mode = m;
            document.querySelectorAll('.seg').forEach(b => b.classList.toggle('active', b === btn));
            document.querySelectorAll('.mode-panel').forEach(p => p.classList.toggle('active', p.id === `mode-${m}`));
        });
    });
}

function setupBulk() {
    $('bulkSubmit').addEventListener('click', () => {
        const text = $('bulkInput').value;
        const parts = text.split(/[\s,;\n]+/).map(s => s.trim()).filter(Boolean);
        let added = 0;
        for (const p of parts) {
            const n = parseInt(p, 10);
            if (!isNaN(n) && n >= 0 && n <= 36) {
                State.rounds.push({ number: n, color: colorOf(n), parity: parityOf(n), time: Date.now() + added });
                added++;
            }
        }
        if (added > 0) {
            save();
            $('bulkInput').value = '';
            renderAll();
            toast(`Added ${added} rounds`, 'success');
        } else {
            toast('No valid numbers found', 'error');
        }
    });
}

function setupUpload() {
    const zone = $('uploadZone');
    const input = $('fileInput');
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', e => { if (e.target.files[0]) importJSON(e.target.files[0]); });
    ['dragenter','dragover'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('dragging'); }));
    ['dragleave','drop'].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('dragging'); }));
    zone.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files[0]) importJSON(e.dataTransfer.files[0]); });
    $('downloadSample').addEventListener('click', downloadSample);
}

function setupPredictBtn() {
    $('predictBtn').addEventListener('click', () => {
        const btn = $('predictBtn');
        btn.disabled = true;
        // Run in next frame to keep UI responsive
        requestAnimationFrame(() => {
            const p = runPrediction();
            btn.disabled = false;
            if (!p) return;
            State.lastPrediction = p;
            State.lastInsight = p.insight;
            renderPrediction(p);
            renderAll();
            const colorLabel  = p.color.charAt(0).toUpperCase() + p.color.slice(1);
            const parityLabel = p.parity === 'zero' ? 'Zero' : (p.parity === 'even' ? 'Even' : 'Odd');
            toast(`${colorLabel} · ${parityLabel}`, 'success');
        });
    });
}

function setupHistoryActions() {
    $('exportBtn').addEventListener('click', exportJSON);
    $('clearBtn').addEventListener('click', () => {
        if (State.rounds.length === 0) { toast('Nothing to clear'); return; }
        showConfirm('Clear all data?', 'This permanently deletes all rounds and predictions.', clearAll);
    });
}

function showConfirm(title, text, onYes) {
    const m = $('confirmModal');
    $('modalTitle').textContent = title;
    $('modalText').textContent = text;
    m.classList.add('show');
    const yes = $('modalConfirm'), no = $('modalCancel');
    const close = () => m.classList.remove('show');
    const yesH = () => { close(); onYes(); cleanup(); };
    const noH  = () => { close(); cleanup(); };
    function cleanup() {
        yes.removeEventListener('click', yesH);
        no.removeEventListener('click', noH);
    }
    yes.addEventListener('click', yesH);
    no.addEventListener('click', noH);
}

function preventZoom() {
    document.addEventListener('gesturestart',  e => e.preventDefault());
    document.addEventListener('gesturechange', e => e.preventDefault());
    document.addEventListener('gestureend',    e => e.preventDefault());
    let lastTouch = 0;
    document.addEventListener('touchend', e => {
        const now = Date.now();
        if (now - lastTouch <= 300) e.preventDefault();
        lastTouch = now;
    }, { passive: false });
    document.addEventListener('touchmove', e => { if (e.touches.length > 1) e.preventDefault(); }, { passive: false });
    document.addEventListener('wheel', e => { if (e.ctrlKey) e.preventDefault(); }, { passive: false });
    document.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && ['+','-','=','0'].includes(e.key)) e.preventDefault();
    });
}

// ==========================================================
//  INIT
// ==========================================================
window.addEventListener('DOMContentLoaded', () => {
    preventZoom();
    load();
    buildNumberGrid();
    setupTabs();
    setupModes();
    setupBulk();
    setupUpload();
    setupPredictBtn();
    setupHistoryActions();
    renderAll();
});

})();
