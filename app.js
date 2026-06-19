/* ==========================================================
   Roulette Predictor — 15-Window edition
   --------------------------------------------------------
   • Active sliding window of 15 rounds (FIFO).
     When the user enters round 16, position 1 drops out,
     positions 2..15 shift to 1..14, and the new entry
     becomes position 15. Prediction is then re-run.
   • Separate, much larger "training data" pool used to
     train pattern lookups (uploaded JSON or bulk paste).
   • 6-model ensemble:
       M1  1st-order Markov chain
       M2  2nd-order Markov chain
       M3  Variable-length n-gram (max length 6)
       M4  15-window similarity vote (cosine on tail)
       M5  EWMA recent frequency
       M6  Streak / anti-streak detector
     Each model gets a weight proportional to its
     walk-forward backtest accuracy on training data.
   --------------------------------------------------------
   HONEST DISCLAIMER: a fair roulette wheel produces
   independent spins. No software can be 100% accurate.
   The ensemble outperforms 50% only if your data has a
   real bias (biased wheel, weak RNG, scripted feed, ...).
   ========================================================== */

(() => {

// European wheel red numbers
const RED = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

// ---------- helpers ----------
const colorOf  = n => n === 0 ? 'green' : (RED.has(n) ? 'red' : 'black');
const parityOf = n => n === 0 ? 'zero' : (n % 2 === 0 ? 'even' : 'odd');

const COLOR_CLASSES  = ['red','black','green'];
const PARITY_CLASSES = ['even','odd','zero'];

const WINDOW_SIZE = 15;

// ---------- state ----------
const State = {
    training: [],       // large pool
    window:   [],       // rolling FIFO, length <= 15
    predictions: [],    // scored predictions log
    lastPrediction: null,
    lastInsight: null,
    tab: 'predict',
    mode: 'tap',
    addDest: 'window',  // where Quick Tap / Bulk go
    histTab: 'window'
};

// ---------- storage ----------
const KEY_T = 'rp_training_v3';
const KEY_W = 'rp_window_v3';
const KEY_P = 'rp_preds_v3';

function save() {
    try {
        localStorage.setItem(KEY_T, JSON.stringify(State.training));
        localStorage.setItem(KEY_W, JSON.stringify(State.window));
        localStorage.setItem(KEY_P, JSON.stringify(State.predictions));
    } catch(_) {}
}
function load() {
    try {
        const t = JSON.parse(localStorage.getItem(KEY_T) || '[]');
        const w = JSON.parse(localStorage.getItem(KEY_W) || '[]');
        const p = JSON.parse(localStorage.getItem(KEY_P) || '[]');
        if (Array.isArray(t)) State.training = t;
        if (Array.isArray(w)) State.window = w.slice(-WINDOW_SIZE);
        if (Array.isArray(p)) State.predictions = p;
    } catch(_) {}

    // Backwards-compat: read v2 keys if v3 empty
    if (State.training.length === 0 && State.window.length === 0) {
        try {
            const v2 = JSON.parse(localStorage.getItem('rp_rounds_v2') || '[]');
            if (Array.isArray(v2) && v2.length) {
                State.window = v2.slice(-WINDOW_SIZE);
                State.training = v2.slice(0, Math.max(0, v2.length - WINDOW_SIZE));
            }
        } catch(_) {}
    }
}

// ==========================================================
// PREDICTION ENGINE — 6-MODEL ENSEMBLE
// ==========================================================

const NGRAM_MAX = 6;
const EWMA_HALFLIFE = 25;
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
function argmax(p, classes) {
    return classes.reduce((a, b) => p[a] >= p[b] ? a : b);
}

// ---- M1: 1st-order Markov ----
function markov1(seq, classes) {
    if (seq.length < 2) return uniform(classes);
    const trans = {};
    classes.forEach(a => { trans[a] = {}; classes.forEach(b => trans[a][b] = LAPLACE_ALPHA); });
    for (let i = 1; i < seq.length; i++) trans[seq[i-1]][seq[i]]++;
    const last = seq[seq.length - 1];
    return normalize({ ...trans[last] }, classes);
}

// ---- M2: 2nd-order Markov ----
function markov2(seq, classes) {
    if (seq.length < 4) return markov1(seq, classes);
    const trans = {}; // key: "a|b" -> counts
    classes.forEach(a => classes.forEach(b => {
        const k = a + '|' + b;
        trans[k] = {};
        classes.forEach(c => trans[k][c] = LAPLACE_ALPHA);
    }));
    for (let i = 2; i < seq.length; i++) {
        const k = seq[i-2] + '|' + seq[i-1];
        trans[k][seq[i]]++;
    }
    const a = seq[seq.length - 2], b = seq[seq.length - 1];
    const k = a + '|' + b;
    return normalize({ ...trans[k] }, classes);
}

// ---- M3: Variable-length n-gram lookup ----
function ngramLookup(seq, classes) {
    if (seq.length < 3) return uniform(classes);
    const scores = {};
    classes.forEach(c => scores[c] = LAPLACE_ALPHA);
    for (let len = Math.min(NGRAM_MAX, seq.length - 1); len >= 2; len--) {
        const pattern = seq.slice(-len);
        let matches = 0;
        for (let i = 0; i <= seq.length - len - 1; i++) {
            let ok = true;
            for (let j = 0; j < len; j++) {
                if (seq[i + j] !== pattern[j]) { ok = false; break; }
            }
            if (ok) {
                const next = seq[i + len];
                scores[next] = (scores[next] || 0) + Math.pow(len, 1.6);
                matches++;
            }
        }
        if (matches >= 3) break;
    }
    return normalize(scores, classes);
}

// ---- M4: 15-window similarity vote ----
// Tail of `seq` is the active window; search the rest for the most
// similar windows (highest fuzzy match on last min(WINDOW_SIZE, len-1)
// symbols). The "next" symbol after each high-similarity window votes,
// weighted by similarity^2.
function windowSimilarity(seq, classes, windowTail) {
    if (seq.length < WINDOW_SIZE + 2 || windowTail.length === 0) return uniform(classes);
    const W = windowTail.length;
    const scores = {};
    classes.forEach(c => scores[c] = LAPLACE_ALPHA);

    // Slide candidate windows across history (excluding the last point,
    // which has no "next" symbol yet)
    for (let i = 0; i <= seq.length - W - 1; i++) {
        let matches = 0;
        for (let j = 0; j < W; j++) {
            if (seq[i + j] === windowTail[j]) matches++;
        }
        const sim = matches / W;             // 0..1
        if (sim < 0.4) continue;             // ignore weak matches
        const next = seq[i + W];
        const weight = Math.pow(sim, 2.0) * (1 + (sim === 1 ? 2 : 0));
        scores[next] = (scores[next] || 0) + weight;
    }
    return normalize(scores, classes);
}

// ---- M5: EWMA recent frequency ----
function ewma(seq, classes) {
    if (seq.length === 0) return uniform(classes);
    const counts = {};
    classes.forEach(c => counts[c] = LAPLACE_ALPHA);
    const n = seq.length;
    const lambda = Math.log(2) / EWMA_HALFLIFE;
    for (let i = 0; i < n; i++) {
        const age = n - 1 - i;
        counts[seq[i]] = (counts[seq[i]] || 0) + Math.exp(-lambda * age);
    }
    return normalize(counts, classes);
}

// ---- M6: Streak / anti-streak ----
// If the last K outcomes are all the same class C, this model
// estimates P(C continues) vs P(switch) using empirical streak
// continuation probability from the training data.
function streakModel(seq, classes) {
    if (seq.length < 4) return uniform(classes);
    // measure current streak
    const last = seq[seq.length - 1];
    let K = 1;
    for (let i = seq.length - 2; i >= 0 && seq[i] === last; i--) K++;
    // gather empirical continuation rates for streaks of length >= K
    let continued = 0, broke = 0;
    let run = 1;
    for (let i = 1; i < seq.length; i++) {
        if (seq[i] === seq[i-1]) {
            run++;
        } else {
            if (run >= K) broke++;
            run = 1;
        }
        if (run >= K && i < seq.length - 1) {
            if (seq[i+1] === seq[i]) continued++;
            else broke++;
        }
    }
    const total = continued + broke;
    const pContinue = total > 0 ? (continued + 1) / (total + 2) : 0.5;
    const scores = {};
    classes.forEach(c => scores[c] = LAPLACE_ALPHA);
    scores[last] += pContinue * 10;
    classes.filter(c => c !== last).forEach(c => scores[c] += (1 - pContinue) * 5);
    return normalize(scores, classes);
}

// ---------- backtest weights ----------
const MODELS = ['m1','m2','m3','m4','m5','m6'];

function backtestWeights(seq, classes, warmup = 16) {
    const hits = { m1:0, m2:0, m3:0, m4:0, m5:0, m6:0 };
    let trials = 0;
    if (seq.length <= warmup + 1) {
        const eq = {};
        MODELS.forEach(m => eq[m] = 1);
        return { ...eq, trials: 0, hits };
    }
    // sample-cap for speed on large training sets
    const cap = 800;
    const startIdx = seq.length - cap > warmup ? seq.length - cap : warmup;
    for (let i = startIdx; i < seq.length; i++) {
        const past = seq.slice(0, i);
        const actual = seq[i];
        const tail = past.slice(-WINDOW_SIZE);
        const p1 = markov1(past, classes);
        const p2 = markov2(past, classes);
        const p3 = ngramLookup(past, classes);
        const p4 = windowSimilarity(past, classes, tail);
        const p5 = ewma(past, classes);
        const p6 = streakModel(past, classes);
        if (argmax(p1, classes) === actual) hits.m1++;
        if (argmax(p2, classes) === actual) hits.m2++;
        if (argmax(p3, classes) === actual) hits.m3++;
        if (argmax(p4, classes) === actual) hits.m4++;
        if (argmax(p5, classes) === actual) hits.m5++;
        if (argmax(p6, classes) === actual) hits.m6++;
        trials++;
    }
    // softmax-ish weighting on accuracy with floor
    const acc = {};
    MODELS.forEach(m => acc[m] = (hits[m] + 1) / (trials + classes.length));
    // amplify gaps: weight = max(0, acc - baseline)^2 + small floor
    const baseline = 1 / classes.length;
    const w = {};
    MODELS.forEach(m => {
        const delta = Math.max(0, acc[m] - baseline);
        w[m] = Math.pow(delta, 2) * 100 + 0.15;   // floor 0.15 so every model contributes a little
    });
    return { ...w, trials, hits };
}

// ---------- main per-kind predict ----------
function predictKind(trainingSeq, windowSeq, classes) {
    // The full sequence used for pattern lookup is training + window
    const full = trainingSeq.concat(windowSeq);

    if (full.length < 2) {
        return { winner: classes[0], probs: uniform(classes), confidence: 0, weights: null, hits: null, trials: 0, perModel: null };
    }
    const tail = windowSeq.slice(-WINDOW_SIZE);
    const p1 = markov1(full, classes);
    const p2 = markov2(full, classes);
    const p3 = ngramLookup(full, classes);
    const p4 = windowSimilarity(full, classes, tail);
    const p5 = ewma(full, classes);
    const p6 = streakModel(full, classes);
    const w = backtestWeights(full, classes);
    const wsum = (w.m1 + w.m2 + w.m3 + w.m4 + w.m5 + w.m6) || 1;

    const blended = {};
    classes.forEach(c => {
        blended[c] = (p1[c]*w.m1 + p2[c]*w.m2 + p3[c]*w.m3 + p4[c]*w.m4 + p5[c]*w.m5 + p6[c]*w.m6) / wsum;
    });

    // winner & calibrated confidence
    const sorted = classes.slice().sort((a, b) => blended[b] - blended[a]);
    const winner = sorted[0];
    const margin = blended[sorted[0]] - blended[sorted[1] || sorted[0]];
    const rawConf = blended[winner] * 100;
    // confidence: weighted top-prob and margin; back-test trials gate the upper bound
    const trialFactor = Math.min(1, w.trials / 100);
    const upper = 60 + 35 * trialFactor;  // cap from 60% with 0 trials → 95% at 100+ trials
    const confidence = Math.max(34, Math.min(upper, Math.round(rawConf * 0.55 + margin * 220)));

    return { winner, probs: blended, confidence, weights: w, hits: w.hits, trials: w.trials,
             perModel: { m1: p1, m2: p2, m3: p3, m4: p4, m5: p5, m6: p6 } };
}

function runPrediction() {
    const wRounds = State.window;
    const tRounds = State.training;
    if (wRounds.length < WINDOW_SIZE) {
        toast(`Window needs ${WINDOW_SIZE - wRounds.length} more round${WINDOW_SIZE - wRounds.length === 1 ? '' : 's'}`, 'error');
        return null;
    }
    const trainingColor  = tRounds.map(x => x.color);
    const trainingParity = tRounds.map(x => x.parity);
    const windowColor    = wRounds.map(x => x.color);
    const windowParity   = wRounds.map(x => x.parity);

    const colorRes  = predictKind(trainingColor,  windowColor,  COLOR_CLASSES);
    const parityRes = predictKind(trainingParity, windowParity, PARITY_CLASSES);

    // Bet-friendly: avoid green/zero as final answer unless dominant
    const adjust = (res, classes, skip) => {
        if (res.winner === skip && res.probs[skip] < 0.45) {
            const alt = classes.filter(c => c !== skip)
                .reduce((a, b) => res.probs[a] >= res.probs[b] ? a : b);
            res.winner = alt;
            res.confidence = Math.max(34, Math.round(res.probs[alt] * 100));
        }
        return res;
    };
    adjust(colorRes,  COLOR_CLASSES,  'green');
    adjust(parityRes, PARITY_CLASSES, 'zero');

    return {
        color: colorRes.winner,
        colorConfidence: colorRes.confidence,
        colorProbs: colorRes.probs,
        parity: parityRes.winner,
        parityConfidence: parityRes.confidence,
        parityProbs: parityRes.probs,
        insight: {
            colorWeights: colorRes.weights, colorHits: colorRes.hits,
            parityWeights: parityRes.weights, parityHits: parityRes.hits,
            trials: colorRes.trials,
            trainingSize: tRounds.length, windowSize: wRounds.length
        }
    };
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
    t._timer = setTimeout(() => t.classList.remove('show'), 2000);
}

function renderBadge() {
    $('trainBadge').textContent  = `T:${State.training.length}`;
    $('windowBadge').textContent = `W:${State.window.length}/${WINDOW_SIZE}`;
    if (State.window.length === WINDOW_SIZE) $('windowBadge').classList.add('full');
    else $('windowBadge').classList.remove('full');
    const a = $('addTrainNum'); if (a) a.textContent = State.training.length;
    const b = $('addWindowNum'); if (b) b.textContent = `${State.window.length}/${WINDOW_SIZE}`;
}

function renderWindowStrip(animateSlide = false) {
    const strip = $('windowStrip');
    if (!strip) return;
    const slots = [];
    for (let i = 0; i < WINDOW_SIZE; i++) {
        const r = State.window[i];
        if (r) {
            slots.push(`
                <div class="slot filled ${r.color}" data-idx="${i}" title="Position ${i + 1}: ${r.number}">
                    <div class="slot-num">${r.number}</div>
                    <div class="slot-pos">#${i + 1}</div>
                </div>`);
        } else {
            slots.push(`<div class="slot empty" data-idx="${i}"><div class="slot-num">·</div><div class="slot-pos">#${i + 1}</div></div>`);
        }
    }
    strip.innerHTML = slots.join('');
    if (animateSlide) {
        strip.classList.remove('slide');
        // force reflow then add the slide class
        void strip.offsetWidth;
        strip.classList.add('slide');
    }
}

function renderPrediction(p) {
    const cv = $('colorValue'), pv = $('parityValue');
    const cb = $('colorConfBar'), pb = $('parityConfBar');
    const ct = $('colorConfText'), pt = $('parityConfText');
    const cp = $('colorProbs'), pp = $('parityProbs');

    if (!p) {
        cv.className = 'pc-value'; cv.textContent = '—';
        pv.className = 'pc-value'; pv.textContent = '—';
        cb.style.width = '0%';     pb.style.width = '0%';
        ct.textContent = '—';      pt.textContent = '—';
        cp.innerHTML = '';         pp.innerHTML = '';
        return;
    }

    const colorLabel = p.color.charAt(0).toUpperCase() + p.color.slice(1);
    cv.className = `pc-value ${p.color}`;
    cv.textContent = colorLabel;
    cb.style.width = `${p.colorConfidence}%`;
    ct.textContent = `${p.colorConfidence}%`;
    cp.innerHTML = COLOR_CLASSES.map(c =>
        `<div class="prob ${c}"><span>${c[0].toUpperCase()}</span><b>${Math.round((p.colorProbs[c]||0)*100)}%</b></div>`
    ).join('');

    const parityLabel = p.parity === 'zero' ? 'Zero' : (p.parity === 'even' ? 'Even' : 'Odd');
    pv.className = `pc-value ${p.parity}`;
    pv.textContent = parityLabel;
    pb.style.width = `${p.parityConfidence}%`;
    pt.textContent = `${p.parityConfidence}%`;
    pp.innerHTML = PARITY_CLASSES.map(c =>
        `<div class="prob ${c}"><span>${c === 'zero' ? '0' : c[0].toUpperCase()}</span><b>${Math.round((p.parityProbs[c]||0)*100)}%</b></div>`
    ).join('');
}

function renderPredictNote() {
    const w = State.window.length;
    const t = State.training.length;
    const el = $('predictNote');
    if (w < WINDOW_SIZE) {
        el.className = 'predict-note warn';
        el.textContent = `Window needs ${WINDOW_SIZE - w} more round${WINDOW_SIZE - w === 1 ? '' : 's'}. Training data: ${t} rounds.`;
    } else if (t < 50) {
        el.className = 'predict-note warn';
        el.textContent = `Window ready ✓ — but training data is small (${t}). Upload more for stronger predictions.`;
    } else {
        el.className = 'predict-note ok';
        el.textContent = `Window ready ✓ · Training on ${t} rounds · 6-model ensemble active.`;
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
    const tab = State.histTab;
    $('histTitle').textContent = tab === 'window' ? 'Window history' : tab === 'training' ? 'Training data' : 'Past predictions';

    if (tab === 'preds') {
        if (State.predictions.length === 0) {
            list.innerHTML = '<p class="empty-state">No graded predictions yet.</p>';
            return;
        }
        const items = [];
        const arr = State.predictions.slice().reverse().slice(0, 200);
        arr.forEach((p, i) => {
            const cOK = p.predColor === p.actColor;
            const oOK = p.predParity === p.actParity;
            const time = p.t ? new Date(p.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
            items.push(`
                <div class="pred-item">
                    <div class="pred-row"><span class="pred-key">Color</span>
                        <span class="pill ${p.predColor}">${p.predColor}</span>
                        <span class="arrow">→</span>
                        <span class="pill ${p.actColor}">${p.actColor}</span>
                        <span class="result ${cOK?'ok':'no'}">${cOK?'✓':'✗'}</span>
                    </div>
                    <div class="pred-row"><span class="pred-key">Parity</span>
                        <span class="pill ${p.predParity}">${p.predParity}</span>
                        <span class="arrow">→</span>
                        <span class="pill ${p.actParity}">${p.actParity}</span>
                        <span class="result ${oOK?'ok':'no'}">${oOK?'✓':'✗'}</span>
                    </div>
                    <div class="pred-meta">${time}</div>
                </div>`);
        });
        list.innerHTML = items.join('');
        return;
    }

    const arr = tab === 'window' ? State.window : State.training;
    if (arr.length === 0) {
        list.innerHTML = '<p class="empty-state">No rounds yet.</p>';
        return;
    }
    const max = 300;
    const len = arr.length;
    const start = Math.max(0, len - max);
    const items = [];
    for (let i = len - 1; i >= start; i--) {
        const r = arr[i];
        const time = r.time ? new Date(r.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
        items.push(`
            <div class="history-item">
                <div class="history-chip ${r.color}">${r.number}</div>
                <div class="history-info">
                    <div class="history-num">${tab==='window'?'Slot':'#'}${i + 1} · ${r.color} · ${r.parity}</div>
                    <div class="history-meta">${time || 'No timestamp'}</div>
                </div>
                <button class="history-delete" data-idx="${i}" data-src="${tab}" aria-label="Delete">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
            </div>`);
    }
    list.innerHTML = items.join('');
    list.querySelectorAll('.history-delete').forEach(b => {
        b.addEventListener('click', () => deleteRound(parseInt(b.dataset.idx, 10), b.dataset.src));
    });
}

function renderStats() {
    const r = State.training.concat(State.window);
    const n = r.length;
    const counts = { red: 0, black: 0, green: 0, even: 0, odd: 0, zero: 0 };
    for (const x of r) { counts[x.color]++; counts[x.parity]++; }
    const pct = c => n ? Math.round(counts[c] / n * 100) : 0;
    const rows = [
        ['red',   'Red'], ['black', 'Black'], ['green', 'Green'],
        ['even',  'Even'], ['odd',   'Odd']
    ];
    $('statRows').innerHTML = rows.map(([key, label]) => `
        <div class="stat-row">
            <div class="stat-key">${label}</div>
            <div class="stat-track"><div class="stat-fill ${key}" style="width:${pct(key)}%"></div></div>
            <div class="stat-value">${counts[key]}<span class="pct">${pct(key)}%</span></div>
        </div>`).join('');

    const freq = {};
    for (const x of r) freq[x.number] = (freq[x.number] || 0) + 1;
    const maxF = Math.max(1, ...Object.values(freq));
    const cells = [];
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

    const ins = State.lastInsight;
    const insightCard = $('insightCard');
    if (!ins) {
        insightCard.innerHTML = '<p class="empty-state">Predict at least once to see model insight.</p>';
        return;
    }
    const cw = ins.colorWeights, pw = ins.parityWeights;
    const ch = ins.colorHits, ph = ins.parityHits;
    const t = ins.trials;
    const sumW = obj => MODELS.reduce((s, m) => s + (obj[m] || 0), 0) || 1;
    const tC = sumW(cw), tP = sumW(pw);
    const modelLabels = {
        m1: 'Markov-1', m2: 'Markov-2', m3: 'N-gram',
        m4: 'Window-sim', m5: 'EWMA', m6: 'Streak'
    };
    const bars = MODELS.map(m => `
        <div class="model-bar">
            <div class="mb-name">${modelLabels[m]}</div>
            <div class="mb-track">
                <div class="mb-fill c" style="width:${(cw[m]/tC*100).toFixed(0)}%" title="color weight"></div>
                <div class="mb-fill p" style="width:${(pw[m]/tP*100).toFixed(0)}%" title="parity weight"></div>
            </div>
            <div class="mb-hits">${ch[m]||0}/${t} · ${ph[m]||0}/${t}</div>
        </div>`).join('');

    insightCard.innerHTML = `
        <div class="ins-row"><span class="ins-key">Training rounds</span><span class="ins-val">${ins.trainingSize}</span></div>
        <div class="ins-row"><span class="ins-key">Window rounds</span><span class="ins-val">${ins.windowSize}/${WINDOW_SIZE}</span></div>
        <div class="ins-row"><span class="ins-key">Backtest trials</span><span class="ins-val">${t}</span></div>
        <div class="model-legend"><span class="leg c">■ Color weight</span><span class="leg p">■ Parity weight</span><span class="leg-meta">Hits: color · parity</span></div>
        <div class="model-bars">${bars}</div>`;
}

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
        renderWindowStrip();
        renderAccuracyStrip();
        renderActiveTab();
    });
}

// ==========================================================
//  ACTIONS
// ==========================================================

function pushToWindow(num, customColor, customTime) {
    const round = {
        number: num,
        color: customColor || colorOf(num),
        parity: parityOf(num),
        time: customTime || Date.now()
    };
    // If full → push oldest into training and shift
    if (State.window.length >= WINDOW_SIZE) {
        const evicted = State.window.shift();
        if (evicted) State.training.push(evicted);
    }
    State.window.push(round);
    return round;
}

function addRoundToWindow(num, customColor = null, customTime = null) {
    num = parseInt(num, 10);
    if (isNaN(num) || num < 0 || num > 36) {
        toast('Number must be 0–36', 'error');
        return false;
    }
    // Score any pending prediction against the new outcome
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
    const wasFull = State.window.length >= WINDOW_SIZE;
    const round = pushToWindow(num, customColor, customTime);
    save();

    // last-added strip
    const la = $('lastAdded');
    if (la) {
        la.classList.add('show');
        la.innerHTML = `<div class="last-chip ${round.color}">${round.number}</div><span>Window slot ${State.window.length} · ${round.color} · ${round.parity}${wasFull ? ' · oldest pushed to training' : ''}</span>`;
    }

    // animate slide if window was full
    renderBadge();
    renderPredictNote();
    renderWindowStrip(wasFull);
    renderAccuracyStrip();
    renderActiveTab();

    // Auto-predict when window just filled or rolled
    if (State.window.length === WINDOW_SIZE) {
        setTimeout(autoPredict, 280);
    }
    return true;
}

function addRoundToTraining(num, customColor = null, customTime = null) {
    num = parseInt(num, 10);
    if (isNaN(num) || num < 0 || num > 36) {
        toast('Number must be 0–36', 'error');
        return false;
    }
    State.training.push({
        number: num,
        color: customColor || colorOf(num),
        parity: parityOf(num),
        time: customTime || Date.now()
    });
    save();
    renderAll();
    return true;
}

function autoPredict() {
    const p = runPrediction();
    if (!p) return;
    State.lastPrediction = p;
    State.lastInsight = p.insight;
    renderPrediction(p);
    renderActiveTab();
    const colorLabel  = p.color.charAt(0).toUpperCase() + p.color.slice(1);
    const parityLabel = p.parity === 'zero' ? 'Zero' : (p.parity === 'even' ? 'Even' : 'Odd');
    toast(`Next: ${colorLabel} · ${parityLabel}`, 'success');
}

function deleteRound(idx, src) {
    if (src === 'window') {
        State.window.splice(idx, 1);
    } else if (src === 'training') {
        State.training.splice(idx, 1);
    }
    save();
    renderAll();
}

function clearAll() {
    const tab = State.histTab;
    if (tab === 'window') {
        State.window = [];
    } else if (tab === 'training') {
        State.training = [];
    } else {
        State.predictions = [];
    }
    State.lastPrediction = null;
    save();
    renderPrediction(null);
    renderAll();
    toast('Cleared');
}

function resetWindow() {
    if (State.window.length === 0) { toast('Window already empty'); return; }
    showConfirm('Reset window?', 'Clears the active 15-slot window. Training data is preserved.', () => {
        State.window = [];
        State.lastPrediction = null;
        save();
        renderPrediction(null);
        renderAll();
        toast('Window reset');
    });
}

function exportJSON() {
    const data = {
        exportedAt: new Date().toISOString(),
        training: State.training,
        window: State.window,
        predictions: State.predictions
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
            else if (Array.isArray(raw.training)) arr = raw.training;
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
                State.training.push({
                    number: parseInt(num, 10),
                    color: customColor || colorOf(parseInt(num, 10)),
                    parity: parityOf(parseInt(num, 10)),
                    time: customTime || (baseTime + i * 1000)
                });
                added++;
            }
            save();
            renderAll();
            toast(`Imported ${added} rounds to training`, 'success');
        } catch (err) {
            console.error(err);
            toast('Could not parse JSON', 'error');
        }
    };
    reader.readAsText(file);
}

function downloadSample() {
    // longer sample so user can backtest
    const nums = [];
    for (let i = 0; i < 120; i++) nums.push(Math.floor(Math.random() * 37));
    const sample = { info: 'Roulette training data sample', rounds: nums };
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

function buildNumberGrids() {
    // Add tab grid
    const grid = $('numberGrid');
    let html = `<button class="num-chip green" data-num="0">0</button>`;
    for (let i = 1; i <= 36; i++) {
        html += `<button class="num-chip ${colorOf(i)}" data-num="${i}">${i}</button>`;
    }
    grid.innerHTML = html;
    grid.addEventListener('click', e => {
        const btn = e.target.closest('.num-chip');
        if (!btn) return;
        const n = parseInt(btn.dataset.num, 10);
        if (State.addDest === 'training') addRoundToTraining(n);
        else addRoundToWindow(n);
    });

    // Predict tab quick-add grid (always goes to window)
    const qa = $('quickAddGrid');
    let qh = `<button class="num-chip green" data-num="0">0</button>`;
    for (let i = 1; i <= 36; i++) {
        qh += `<button class="num-chip ${colorOf(i)}" data-num="${i}">${i}</button>`;
    }
    qa.innerHTML = qh;
    qa.addEventListener('click', e => {
        const btn = e.target.closest('.num-chip');
        if (!btn) return;
        addRoundToWindow(parseInt(btn.dataset.num, 10));
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
    document.querySelectorAll('.seg[data-mode]').forEach(btn => {
        btn.addEventListener('click', () => {
            const m = btn.dataset.mode;
            if (State.mode === m) return;
            State.mode = m;
            document.querySelectorAll('.seg[data-mode]').forEach(b => b.classList.toggle('active', b === btn));
            document.querySelectorAll('.mode-panel').forEach(p => p.classList.toggle('active', p.id === `mode-${m}`));
        });
    });

    // Dest toggles
    document.querySelectorAll('.dest-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            State.addDest = btn.dataset.dest;
            document.querySelectorAll('.dest-btn').forEach(b => b.classList.toggle('active', b.dataset.dest === State.addDest));
        });
    });
    document.querySelectorAll('.dest-btn-bulk').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.dest-btn-bulk').forEach(b => b.classList.toggle('active', b === btn));
            btn._dest = btn.dataset.dest;
        });
    });

    // History sub-tabs
    document.querySelectorAll('.seg[data-htab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const t = btn.dataset.htab;
            State.histTab = t;
            document.querySelectorAll('.seg[data-htab]').forEach(b => b.classList.toggle('active', b === btn));
            renderHistory();
        });
    });
}

function setupBulk() {
    $('bulkSubmit').addEventListener('click', () => {
        const dest = document.querySelector('.dest-btn-bulk.active')?.dataset.dest || 'window';
        const text = $('bulkInput').value;
        const parts = text.split(/[\s,;\n]+/).map(s => s.trim()).filter(Boolean);
        const nums = [];
        for (const p of parts) {
            const n = parseInt(p, 10);
            if (!isNaN(n) && n >= 0 && n <= 36) nums.push(n);
        }
        if (nums.length === 0) { toast('No valid numbers found', 'error'); return; }

        if (dest === 'window') {
            // last 15 win; older ones flow into training
            const split = Math.max(0, nums.length - WINDOW_SIZE);
            for (let i = 0; i < split; i++) {
                const n = nums[i];
                State.training.push({ number: n, color: colorOf(n), parity: parityOf(n), time: Date.now() + i });
            }
            // For window, also drain existing window's old entries when needed
            for (let i = split; i < nums.length; i++) {
                const n = nums[i];
                if (State.window.length >= WINDOW_SIZE) {
                    State.training.push(State.window.shift());
                }
                State.window.push({ number: n, color: colorOf(n), parity: parityOf(n), time: Date.now() + i });
            }
            save();
            $('bulkInput').value = '';
            renderAll();
            toast(`Added ${nums.length} → window`, 'success');
            if (State.window.length === WINDOW_SIZE) setTimeout(autoPredict, 280);
        } else {
            for (let i = 0; i < nums.length; i++) {
                const n = nums[i];
                State.training.push({ number: n, color: colorOf(n), parity: parityOf(n), time: Date.now() + i });
            }
            save();
            $('bulkInput').value = '';
            renderAll();
            toast(`Added ${nums.length} → training`, 'success');
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
        requestAnimationFrame(() => {
            const p = runPrediction();
            btn.disabled = false;
            if (!p) return;
            State.lastPrediction = p;
            State.lastInsight = p.insight;
            renderPrediction(p);
            renderActiveTab();
            const colorLabel  = p.color.charAt(0).toUpperCase() + p.color.slice(1);
            const parityLabel = p.parity === 'zero' ? 'Zero' : (p.parity === 'even' ? 'Even' : 'Odd');
            toast(`${colorLabel} · ${parityLabel}`, 'success');
        });
    });

    $('resetWindowBtn').addEventListener('click', resetWindow);
}

function setupHistoryActions() {
    $('exportBtn').addEventListener('click', exportJSON);
    $('clearBtn').addEventListener('click', () => {
        const tab = State.histTab;
        const arr = tab === 'window' ? State.window : tab === 'training' ? State.training : State.predictions;
        if (arr.length === 0) { toast('Nothing to clear'); return; }
        const labelMap = { window: 'window', training: 'training data', preds: 'predictions log' };
        showConfirm(`Clear ${labelMap[tab]}?`, `This permanently deletes the ${labelMap[tab]}.`, clearAll);
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
    buildNumberGrids();
    setupTabs();
    setupModes();
    setupBulk();
    setupUpload();
    setupPredictBtn();
    setupHistoryActions();
    renderAll();
});

})();
