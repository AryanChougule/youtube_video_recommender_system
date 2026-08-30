/* ReelRank UI.
   Vanilla JS on purpose: no build step, no bundle, no framework version to rot.
   The whole client is ~400 lines and the deployed image stays tiny.

   State model: watch history lives ONLY in the browser and is posted with each
   request. The server is stateless, so a refresh loses nothing that two clicks
   cannot rebuild, and the container scales horizontally. */

const state = {
  history: [],          // [{video_id, title, category, weight}]
  personas: [],
  activePersona: null,
  category: null,
  query: null,
  meta: null,
  lastResponse: null,
  objectives: {},        // task -> weight, live in the Recommendation Lab
  intentScale: 0,
};

// Presets for the Recommendation Lab. These are product stances, not tuned
// optima: "what should this system be FOR?" is a decision, and making it a
// one-click decision is the point of the panel.
const OBJECTIVE_PRESETS = {
  balanced:     null,                       // whatever config.yaml ships
  engagement:   { click: 1.0, long_watch: 0.3, completion: 0, liked: 0, satisfied: 0, dismissed: 0 },
  satisfaction: { click: 0, long_watch: 0.15, completion: 0.25, liked: 0.2, satisfied: 1.0, dismissed: -0.4 },
  discovery:    { click: 0.1, long_watch: 0.2, completion: 0.3, liked: 0.3, satisfied: 0.6, dismissed: -0.2 },
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ utils */

// Deterministic colour from a video id, so a given video always renders the
// same "thumbnail". Real YouTube thumbnails are absent from the synthetic
// catalog; when a real thumbnail_url exists we use it instead.
function hashCode(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  return Math.abs(h);
}
function gradientFor(id) {
  const h = hashCode(id);
  const a = h % 360, b = (a + 40 + (h % 60)) % 360;
  return `linear-gradient(135deg, hsl(${a} 62% 34%), hsl(${b} 58% 20%))`;
}
function avatarColor(name) {
  const h = hashCode(name || 'x') % 360;
  return `hsl(${h} 48% 38%)`;
}
function fmtViews(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B views';
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M views';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K views';
  return n + ' views';
}
function fmtDuration(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const pad = (x) => String(x).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}
function fmtAge(days) {
  if (days < 1) return 'today';
  if (days < 30) return `${Math.round(days)} days ago`;
  if (days < 365) return `${Math.round(days / 30)} months ago`;
  return `${(days / 365).toFixed(1)} years ago`;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

/* ------------------------------------------------------------- rendering */

function thumbHTML(item, rank) {
  const inner = item.thumbnail_url
    ? `<img src="${esc(item.thumbnail_url)}" alt="" loading="lazy">`
    : `<div class="thumb-label">${esc((item.category || '?').split(/[\s&]+/)[0])}</div>`;
  const badge = rank !== undefined ? `<span class="rank-badge">#${rank + 1}</span>` : '';
  return `<div class="thumb" style="background:${gradientFor(item.video_id)}">
    ${inner}${badge}
    <span class="duration">${fmtDuration(item.duration_seconds)}</span>
  </div>`;
}

function cardHTML(item, index) {
  const initial = (item.channel_title || '?').replace(/^The /, '')[0].toUpperCase();
  const tags = Object.keys(item.sources || {})
    .map((s) => `<span class="src-tag ${esc(s)}">${esc(s)}</span>`).join('');
  const isExplore = (item.policy_notes || []).includes('exploration');
  const why = item.explanation
    ? `<div class="why ${isExplore ? 'explore' : ''}" data-why="${index}">
         ${isExplore ? '&#10023; ' : '&#128161; '}${esc(item.explanation)}
       </div>` : '';
  return `<article class="card" data-idx="${index}">
    ${thumbHTML(item, item.rank)}
    <div class="card-body">
      <div class="avatar" style="background:${avatarColor(item.channel_title)}">${esc(initial)}</div>
      <div style="min-width:0;flex:1">
        <h3 class="card-title">${esc(item.title)}</h3>
        <div class="card-meta">${esc(item.channel_title)}</div>
        <div class="card-meta">${fmtViews(item.view_count)} &middot; ${fmtAge(item.age_days)}</div>
        ${why}
        <div class="src-tags">${tags}</div>
      </div>
    </div>
  </article>`;
}

function renderFeed(items, title, subtitle) {
  $('feed-title').textContent = title;
  $('feed-sub').textContent = subtitle || '';
  if (!items.length) {
    $('feed').innerHTML = '<div class="loading">No results.</div>';
    return;
  }
  $('feed').innerHTML = `<div class="grid">${items.map(cardHTML).join('')}</div>`;
  $('feed').querySelectorAll('.card').forEach((el) => {
    el.addEventListener('click', (ev) => {
      const idx = +el.dataset.idx;
      if (ev.target.closest('.why')) { openExplain(idx); return; }
      openVideo(items[idx]);
    });
  });
}

function renderPersonas() {
  $('personas').innerHTML = state.personas.map((p) => `
    <button class="persona ${state.activePersona === p.key ? 'active' : ''}" data-key="${esc(p.key)}">
      <div class="persona-name">${esc(p.name)}</div>
      <div class="persona-desc">${esc(p.description)}</div>
    </button>`).join('');
  $('personas').querySelectorAll('.persona').forEach((el) => {
    el.addEventListener('click', () => applyPersona(el.dataset.key));
  });
}

function renderHistory() {
  $('hist-count').textContent = state.history.length ? `(${state.history.length})` : '';
  if (!state.history.length) {
    $('history').innerHTML = '<div class="empty-note">Empty &mdash; the engine will use the cold-start path (trending + popular).</div>';
    return;
  }
  $('history').innerHTML = state.history.slice().reverse().map((h) => `
    <div class="hist-item">
      <div class="hist-thumb" style="background:${gradientFor(h.video_id)}"></div>
      <div class="hist-title" title="${esc(h.title)}">${esc(h.title)}</div>
      <button class="hist-remove" data-id="${esc(h.video_id)}" title="Remove">&times;</button>
    </div>`).join('');
  $('history').querySelectorAll('.hist-remove').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.stopPropagation();
      state.history = state.history.filter((h) => h.video_id !== el.dataset.id);
      state.activePersona = null;
      renderPersonas(); renderHistory(); refresh();
    });
  });
}

function renderObjectives() {
  const weights = state.objectives;
  const names = Object.keys(weights);
  if (!names.length) { $('objectives').innerHTML = ''; return; }
  $('objectives').innerHTML = names.map((k) => `
    <div class="obj-row ${weights[k] < 0 ? 'negative' : ''}">
      <label>${esc(k.replace(/_/g, ' '))} <b>${weights[k].toFixed(2)}</b></label>
      <input type="range" data-obj="${esc(k)}" min="${k === 'dismissed' ? -1 : 0}"
             max="1" step="0.05" value="${weights[k]}">
    </div>`).join('');
  $('objectives').querySelectorAll('input[type=range]').forEach((el) => {
    el.addEventListener('input', () => {
      state.objectives[el.dataset.obj] = +el.value;
      el.parentElement.querySelector('b').textContent = (+el.value).toFixed(2);
      el.parentElement.classList.toggle('negative', +el.value < 0);
    });
    el.addEventListener('change', refresh);
  });
}

function renderIntent(res) {
  const si = (res.diagnostics || {}).session_intent;
  const box = $('intent-box');
  if (!si || !si.detected) {
    box.innerHTML = `<span class="muted">No focused session intent detected
      &mdash; the feed is driven by the long-term profile.</span>`;
    return;
  }
  box.innerHTML = `&#9673; Current focus: <b>${esc(si.label || 'unnamed')}</b><br>
    <span class="muted">coherence ${si.coherence} &middot; novelty ${si.novelty}
    &middot; blend weight applied ${res.diagnostics.intent_applied}</span>`;
}

function renderDiagnostics(res) {
  const s = res.stages || {}, d = res.diagnostics || {};
  const total = s.total_ms || 1;
  const stage = (label, ms, color) => `
    <div class="diag-row"><span>${label}</span><span>${(ms || 0).toFixed(1)} ms</span></div>
    <div class="bar"><div class="bar-fill" style="width:${Math.min(100, (ms / total) * 100)}%;background:${color}"></div></div>`;
  const srcRows = Object.entries(s.sources || {})
    .map(([k, v]) => `<div class="diag-row"><span>&nbsp;&nbsp;${k}</span><span>${v}</span></div>`).join('');
  $('diag').innerHTML = `
    ${stage('Stage 1 &middot; recall', s.recall_ms, '#3ea6ff')}
    ${stage('Stage 2 &middot; ranking', s.rank_ms, '#ffb300')}
    ${stage('Stage 3 &middot; policy', s.policy_ms, '#b388ff')}
    <div class="diag-row" style="margin-top:6px"><span><b>Total</b></span><span><b>${(s.total_ms || 0).toFixed(1)} ms</b></span></div>
    <div class="diag-row"><span>Catalog</span><span>${(s.catalog_size || 0).toLocaleString()}</span></div>
    <div class="diag-row"><span>Candidates kept</span><span>${s.n_candidates || 0}</span></div>
    ${srcRows}
    <div class="diag-row" style="margin-top:8px"><span>Intra-list diversity</span><span>${d.intra_list_diversity ?? '-'}</span></div>
    <div class="diag-row"><span>Novelty (bits)</span><span>${d.novelty_bits ?? '-'}</span></div>
    <div class="diag-row"><span>Distinct categories</span><span>${d.distinct_categories ?? '-'}</span></div>
    <div class="diag-row"><span>Distinct channels</span><span>${d.distinct_channels ?? '-'}</span></div>
    <div class="diag-row"><span>Median age</span><span>${d.median_age_days ?? '-'} d</span></div>`;
}

/* ---------------------------------------------------------------- modals */

function objectivesHTML(obj) {
  if (!obj || !obj.probabilities) return '';
  const names = Object.keys(obj.probabilities);
  // Sort by absolute contribution: the reader wants to know what actually
  // drove the score, not the declaration order of the heads.
  names.sort((a, b) => Math.abs(obj.contributions[b]) - Math.abs(obj.contributions[a]));
  const peak = Math.max(...names.map((k) => Math.abs(obj.contributions[k]))) || 1;
  const rows = names.map((k) => {
    const p = obj.probabilities[k], w = obj.weights[k] ?? 0, c = obj.contributions[k];
    const pct = Math.round((Math.abs(c) / peak) * 100);
    const neg = c < 0;
    return `<tr>
      <td>${esc(k.replace(/_/g, ' '))}</td>
      <td class="num">${(p * 100).toFixed(1)}%</td>
      <td class="num ${w < 0 ? 'neg' : ''}">${w >= 0 ? '+' : ''}${w.toFixed(2)}</td>
      <td class="num ${neg ? 'neg' : ''}"><b>${c >= 0 ? '+' : ''}${c.toFixed(4)}</b></td>
      <td class="barcell"><span class="minibar ${neg ? 'neg' : ''}" style="width:${pct}%"></span></td>
    </tr>`;
  }).join('');
  return `
    <h4>Stage 2 &mdash; objective breakdown</h4>
    <p class="obj-note">Six calibrated heads over one shared feature matrix.
      The value score is <code>&Sigma; weight &times; P(outcome)</code>; the
      weights are live in the Recommendation Lab.</p>
    <table class="obj-table">
      <thead><tr><th>objective</th><th class="num">P</th><th class="num">weight</th>
        <th class="num">contribution</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><td colspan="3">value score</td>
        <td class="num"><b>${obj.total.toFixed(4)}</b></td><td></td></tr></tfoot>
    </table>`;
}

function openExplain(idx) {
  const item = state.lastResponse.items[idx];
  const d = item.explanation_detail || {};
  const rows = Object.entries(item.sources || {}).map(([src, rank]) => `
    <div class="diag-row"><span>${esc(src)}</span>
      <span>rank #${rank + 1} &middot; score ${item.source_scores[src] ?? '-'}</span></div>`).join('');
  $('modal-body').innerHTML = `
    <h2>Why this video?</h2>
    <p style="color:var(--text-dim);margin:0 0 16px">${esc(item.title)}</p>
    <div class="banner" style="background:rgba(62,166,255,.08);border-color:#1c4c73;color:#a9d6ff">
      ${esc(item.explanation)}
    </div>
    <h4>Stage 1 &mdash; which recall sources proposed it</h4>
    <div class="diag">${rows || '<div class="empty-note">No source data.</div>'}</div>
    ${objectivesHTML(d.objectives)}
    <h4>Stage 2 &mdash; ranker</h4>
    <div class="diag">
      <div class="diag-row"><span>Ranker score (expected-watch-time odds)</span><span>${item.ranker_score}</span></div>
      <div class="diag-row"><span>Final score after policy</span><span>${item.score}</span></div>
      <div class="diag-row"><span>Final position</span><span>#${item.rank + 1}</span></div>
    </div>
    <h4>Stage 3 &mdash; policy actions</h4>
    <div class="diag">${(item.policy_notes || []).length
      ? item.policy_notes.map((n) => `<div class="diag-row"><span>${esc(n)}</span><span>applied</span></div>`).join('')
      : '<div class="empty-note">No policy adjustment; ranked purely on relevance.</div>'}</div>
    <h4>Raw explanation payload</h4>
    <pre class="json">${esc(JSON.stringify(d, null, 2))}</pre>`;
  $('modal').classList.add('open');
}

async function openVideo(item) {
  $('modal-body').innerHTML = '<div class="loading"><div class="spinner"></div>Loading&hellip;</div>';
  $('modal').classList.add('open');
  const [video, similar] = await Promise.all([
    api(`/api/video/${item.video_id}`),
    api(`/api/similar/${item.video_id}?n=8`),
  ]);
  const tags = (video.tags || []).filter(Boolean)
    .map((t) => `<span class="tag-pill">${esc(t)}</span>`).join('');
  $('modal-body').innerHTML = `
    <div class="player" style="background:${gradientFor(video.video_id)}">
      <div class="thumb-label">${esc(video.category)}</div>
    </div>
    <h2>${esc(video.title)}</h2>
    <p style="color:var(--text-dim);margin:0 0 14px">
      ${esc(video.channel_title)} &middot; ${fmtViews(video.view_count)} &middot;
      ${fmtAge(video.age_days)} &middot; ${fmtDuration(video.duration_seconds)}
    </p>
    <button class="btn btn-primary" id="btn-watch">&#9654; Watch this (add to history)</button>
    <h4>Metadata the model sees</h4>
    <dl class="kv">
      <dt>Category</dt><dd>${esc(video.category)}</dd>
      <dt>Likes / comments</dt><dd>${video.like_count.toLocaleString()} / ${video.comment_count.toLocaleString()}</dd>
      <dt>Engagement rate</dt><dd>${(video.like_count / Math.max(video.view_count, 1) * 100).toFixed(2)}%</dd>
      <dt>Published</dt><dd>${esc(video.published_at)}</dd>
      <dt>Tags</dt><dd>${tags || '<em>none</em>'}</dd>
    </dl>
    <h4>Up next &mdash; "more like this" (watch-page rail, no personalisation)</h4>
    <div class="grid" id="similar-grid">
      ${similar.items.map((s, i) => cardHTML(s, i)).join('')}
    </div>`;
  $('btn-watch').addEventListener('click', () => {
    addToHistory(video);
    $('modal').classList.remove('open');
  });
  $('similar-grid').querySelectorAll('.card').forEach((el) => {
    el.addEventListener('click', () => openVideo(similar.items[+el.dataset.idx]));
  });
}

function renderHow() {
  const m = state.meta || {};
  const ev = (m.evaluation || {}).reranking_logged_impressions || {};
  const rows = Object.entries(ev)
    .sort((a, b) => b[1].top1 - a[1].top1)
    .map(([k, v]) => `<div class="diag-row"><span>${esc(k)}</span><span>top-1 ${v.top1} &middot; NDCG ${v.ndcg}</span></div>`)
    .join('');
  $('how-body').innerHTML = `
    <h2>How ReelRank works</h2>
    <p style="color:var(--text-dim)">A three-stage pipeline, the same shape YouTube described in
      <em>Deep Neural Networks for YouTube Recommendations</em> (RecSys 2016).</p>

    <h4>Stage 1 &mdash; candidate generation (cheap, high recall)</h4>
    <p style="font-size:13px;color:var(--text-dim)">
      Five retrievers run in parallel over ${(m.catalog || {}).size?.toLocaleString?.() || '?'} videos and are merged with
      Reciprocal Rank Fusion: content embeddings (${esc((m.models || {}).text_backend || '')},
      ${(m.models || {}).text_dims}-d), item-item co-visitation, implicit-feedback ALS
      (${(m.models || {}).als_factors} factors), channel affinity, and trending.
      Scores from different retrievers are not comparable, so RRF fuses on RANK, not score.
    </p>
    <h4>Stage 2 &mdash; ranking (expensive, high precision)</h4>
    <p style="font-size:13px;color:var(--text-dim)">
      A gradient-boosted model over ${(m.models || {}).ranker_features} features scores the ~400 survivors.
      Positives are weighted by watch time, so the model's ODDS estimate expected watch
      time rather than click probability &mdash; the fix YouTube shipped after
      click-optimised ranking produced clickbait.
    </p>
    <h4>Stage 3 &mdash; policy (what makes a good PAGE)</h4>
    <p style="font-size:13px;color:var(--text-dim)">
      MMR diversity (&lambda;=${(m.policy || {}).mmr_lambda}), a hard cap of
      ${(m.policy || {}).max_per_channel} per channel, a freshness boost, and
      ${(m.policy || {}).exploration_slots} reserved exploration slots. Adjust all of these live in the sidebar.
    </p>

    <h4>Measured quality (re-ranking logged impressions)</h4>
    <p style="font-size:12.5px;color:var(--text-dimmer);margin-top:-4px">
      Counterfactually valid: only re-orders videos users were actually shown.
      "[ref] shown position" is not a competing model &mdash; position <em>causes</em> clicks,
      so it measures position bias, not quality.
    </p>
    <div class="diag">${rows || '<div class="empty-note">Run scripts/05_evaluate.py to populate.</div>'}</div>

    <h4>Important caveat</h4>
    <div class="banner">
      <b>Interactions are simulated.</b> No public dataset of real YouTube watch
      histories exists, so user behaviour comes from an explicit, documented
      simulator (cascade click model, position bias, popularity bias, taste drift).
      Video metadata is ${esc((m.catalog || {}).source || '?')}. These metrics show that the algorithms
      recover the latent structure that generated the data &mdash; they are not a
      prediction of real-world YouTube performance.
    </div>`;
  $('how-modal').classList.add('open');
}

/* -------------------------------------------------------------- actions */

function addToHistory(video) {
  if (state.history.some((h) => h.video_id === video.video_id)) return;
  state.history.push({
    video_id: video.video_id, title: video.title,
    category: video.category, weight: 0.7,
  });
  state.activePersona = null;
  renderPersonas(); renderHistory(); refresh();
}

async function applyPersona(key) {
  const persona = state.personas.find((p) => p.key === key);
  if (!persona) return;
  state.activePersona = key;
  state.query = null;
  $('search-input').value = '';
  state.history = persona.videos.map((v) => ({
    video_id: v.video_id, title: v.title, category: v.category, weight: 0.7,
  }));
  renderPersonas(); renderHistory(); refresh();
}

async function refresh() {
  $('feed').innerHTML = '<div class="loading"><div class="spinner"></div>Running the pipeline&hellip;</div>';
  const body = {
    history: state.history.map((h) => h.video_id),
    watch_weights: state.history.map((h) => h.weight),
    n: 24,
    mmr_lambda: +$('mmr').value,
    exploration_slots: +$('explore').value,
    max_per_channel: +$('chan').value,
  };
  if (state.query) body.query = state.query;
  if (Object.keys(state.objectives).length) body.objective_weights = state.objectives;
  body.intent_alpha_scale = state.intentScale;

  try {
    const res = await api('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    state.lastResponse = res;
    let items = res.items;
    if (state.category) items = items.filter((i) => i.category === state.category);

    const mode = res.request.mode;
    const titles = {
      cold_start: 'Trending now',
      personalised: 'Recommended for you',
      search: `Results for "${state.query}"`,
      watch_page: 'Up next',
    };
    const sub = `${mode.replace('_', ' ')} &middot; ${res.stages.n_candidates} candidates `
      + `&rarr; ${res.items.length} shown &middot; ${res.stages.total_ms} ms`;
    $('feed-sub').innerHTML = sub;
    renderFeed(items, titles[mode] || 'Recommended', '');
    $('feed-sub').innerHTML = sub;
    renderDiagnostics(res);
    renderIntent(res);
  } catch (err) {
    $('feed').innerHTML = `<div class="loading">Request failed: ${esc(err.message)}</div>`;
  }
}

function renderChips(categories) {
  const all = [null, ...categories];
  $('chips').innerHTML = all.map((c) => `
    <button class="chip ${state.category === c ? 'active' : ''}" data-cat="${c === null ? '' : esc(c)}">
      ${c === null ? 'All' : esc(c)}
    </button>`).join('');
  $('chips').querySelectorAll('.chip').forEach((el) => {
    el.addEventListener('click', () => {
      state.category = el.dataset.cat || null;
      renderChips(categories);
      if (state.lastResponse) {
        let items = state.lastResponse.items;
        if (state.category) items = items.filter((i) => i.category === state.category);
        renderFeed(items, $('feed-title').textContent, '');
      }
    });
  });
}

/* ------------------------------------------------------------------ init */

function bindControls() {
  [['mmr', 'v-mmr'], ['explore', 'v-explore'], ['chan', 'v-chan']].forEach(([id, out]) => {
    const el = $(id);
    el.addEventListener('input', () => { $(out).textContent = el.value; });
    el.addEventListener('change', refresh);
  });
  $('btn-clear').addEventListener('click', () => {
    state.history = []; state.activePersona = null; state.query = null;
    $('search-input').value = '';
    renderPersonas(); renderHistory(); refresh();
  });
  $('search-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    state.query = $('search-input').value.trim() || null;
    refresh();
  });
  $('intent').addEventListener('input', () => {
    state.intentScale = +$('intent').value;
    $('v-intent').textContent = state.intentScale.toFixed(2);
  });
  $('intent').addEventListener('change', refresh);
  document.querySelectorAll('[data-preset]').forEach((el) => {
    el.addEventListener('click', () => {
      document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('active'));
      el.classList.add('active');
      const preset = OBJECTIVE_PRESETS[el.dataset.preset];
      state.objectives = { ...(preset || state.meta.objectives.default_weights) };
      renderObjectives();
      refresh();
    });
  });
  $('btn-how').addEventListener('click', renderHow);
  ['modal', 'how-modal'].forEach((id) => {
    $(id).addEventListener('click', (ev) => {
      if (ev.target === $(id)) $(id).classList.remove('open');
    });
  });
  $('modal-close').addEventListener('click', () => $('modal').classList.remove('open'));
  $('how-close').addEventListener('click', () => $('how-modal').classList.remove('open'));
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') document.querySelectorAll('.modal-backdrop.open')
      .forEach((m) => m.classList.remove('open'));
  });
}

(async function init() {
  bindControls();
  try {
    const health = await api('/api/health');
    if (!health.ready) {
      $('feed').innerHTML = `<div class="loading">Artifacts not loaded.<br><br>
        <code>${esc(health.error || 'unknown error')}</code></div>`;
      return;
    }
    const [meta, personas] = await Promise.all([api('/api/meta'), api('/api/personas')]);
    state.meta = meta;
    state.personas = personas.personas;

    $('banner-slot').innerHTML = `<div class="banner">
      <b>Interactions are simulated.</b> No public dataset of real YouTube watch histories exists,
      so behaviour comes from a documented simulator; video metadata is <b>${esc(meta.catalog.source)}</b>.
      ${meta.data.n_users?.toLocaleString?.() || '?'} users &middot;
      ${meta.data.n_clicks?.toLocaleString?.() || '?'} clicks &middot;
      ${meta.catalog.size.toLocaleString()} videos.
      <a href="#" id="banner-more">Read the full caveat &rarr;</a>
    </div>`;
    $('banner-more').addEventListener('click', (e) => { e.preventDefault(); renderHow(); });

    if (meta.objectives && meta.objectives.multitask_trained) {
      state.objectives = { ...meta.objectives.default_weights };
      renderObjectives();
      document.querySelector('[data-preset="balanced"]').classList.add('active');
    } else {
      $('objectives').innerHTML =
        '<div class="empty-note">Multi-objective heads not trained.</div>';
    }
    renderChips(meta.catalog.categories);
    renderPersonas();
    renderHistory();
    await refresh();
  } catch (err) {
    $('feed').innerHTML = `<div class="loading">Could not reach the API: ${esc(err.message)}</div>`;
  }
})();
