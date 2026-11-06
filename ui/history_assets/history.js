// Past Generations: a small app inside a Gradio 6 gr.HTML. `element`, `trigger` and `server` come from Gradio;
// GH_CSS is prepended by ui/history_tab.py. Filtering, sorting and rendering run in the browser; the server
// functions (direct POSTs, outside the queue) index the outputs folder, make thumbnails and act on files.
(() => {
  'use strict';
  const host = element.querySelector('.gh-mount') || element;
  if (host.dataset.ghInit === '1') return;
  host.dataset.ghInit = '1';
  if (!document.getElementById('gh-style')) {
    const st = document.createElement('style');
    st.id = 'gh-style';
    st.textContent = GH_CSS;
    document.head.appendChild(st);
  }

  // ------------------------------------------------------------------ helpers
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const U = (u) => u || '';
  const LS = {
    get(k, d) { try { const v = localStorage.getItem('gh_' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('gh_' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
  const fmtSize = (b) => {
    if (!b) return '';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0; let v = b;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)) + ' ' + u[i];
  };
  const fmtDur = (s) => {
    if (s == null || !isFinite(s)) return '';
    s = Math.max(0, s);
    const h = Math.floor(s / 3600); const m = Math.floor((s % 3600) / 60); const sec = s % 60;
    const ss = (m || h) ? Math.floor(sec).toString().padStart(2, '0') : sec.toFixed(sec < 10 ? 1 : 0);
    return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : (m ? `${m}:${ss}` : `${ss}s`);
  };
  const fmtClock = (s) => {
    if (!isFinite(s)) return '0:00.00';
    const m = Math.floor(s / 60); const r = s - m * 60;
    return `${m}:${r.toFixed(2).padStart(5, '0')}`;
  };
  const fmtRes = (w, h) => (w && h ? `${w}×${h}` : '');
  const fmtElapsed = (s) => (s == null ? '' : s < 60 ? `${s.toFixed(1)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`);
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const hhmm = (d) => `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  const dayStart = (d = new Date()) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime() / 1000;
  const relTime = (ts) => {
    if (!ts) return '';
    const now = Date.now() / 1000; const diff = now - ts; const d = new Date(ts * 1000);
    if (diff < 45) return 'just now';
    if (diff < 3600) return `${Math.max(1, Math.round(diff / 60))} min ago`;
    if (ts >= dayStart()) return `${Math.round(diff / 3600)} h ago`;
    if (ts >= dayStart() - 86400) return `Yesterday ${hhmm(d)}`;
    if (diff < 6 * 86400) return `${DAYS[d.getDay()].slice(0, 3)} ${hhmm(d)}`;
    return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
  };
  const fullDate = (ts) => {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    return `${DAYS[d.getDay()]}, ${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()} · ${hhmm(d)}`;
  };
  const bucketOf = (ts) => {
    const t0 = dayStart();
    if (ts >= t0) return ['0', 'Today'];
    if (ts >= t0 - 86400) return ['1', 'Yesterday'];
    if (ts >= t0 - 6 * 86400) { const d = new Date(ts * 1000); return ['2' + d.getDay(), DAYS[d.getDay()]]; }
    const d = new Date(ts * 1000);
    return [`m${d.getFullYear()}${String(d.getMonth()).padStart(2, '0')}`, `${['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][d.getMonth()]} ${d.getFullYear()}`];
  };
  const isTyping = (t) => t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
  const statusName = { ok: 'Finished', failed: 'Failed', cancelled: 'Cancelled', partial: 'Partial', incomplete: 'Unfinished' };

  async function call(fn, data) {
    try {
      return await server[fn](data);
    } catch (e) {
      console.error('[Past Generations]', fn, e);
      toast('⚠️ ' + ((e && e.message) || 'The server did not answer.'), 'bad');
      return null;
    }
  }

  // ------------------------------------------------------------------ toasts, menu, dialog (under <body>)
  const portal = (cls, html = '') => { const el = document.createElement('div'); el.className = cls; el.innerHTML = html; document.body.appendChild(el); return el; };
  const toastBox = portal('gh-toasts');
  function toast(msg, kind = '', ms = 3200) {
    const t = document.createElement('div');
    t.className = 'gh-toast ' + kind;
    t.textContent = msg;
    toastBox.appendChild(t);
    setTimeout(() => { t.style.transition = 'opacity .3s'; t.style.opacity = '0'; setTimeout(() => t.remove(), 320); }, ms);
  }
  const menu = portal('gh-menu');
  function openMenu(anchor, title, entries) {
    menu.innerHTML = `<div class="title">${esc(title)}</div>` + entries.map((e, i) => `<button data-i="${i}"><span class="dot" style="--c:${e.color}"></span>${esc(e.icon)} ${esc(e.label)}</button>`).join('');
    menu.classList.add('show');
    const r = anchor.getBoundingClientRect();
    const mh = menu.offsetHeight; const mw = menu.offsetWidth;
    let top = r.bottom + 6; if (top + mh > innerHeight - 8) top = Math.max(8, r.top - mh - 6);
    menu.style.top = top + 'px';
    menu.style.left = Math.min(innerWidth - mw - 8, Math.max(8, r.left)) + 'px';
    menu.onclick = (ev) => { const b = ev.target.closest('button'); if (!b) return; closeMenu(); entries[+b.dataset.i].run(); };
  }
  function closeMenu() { menu.classList.remove('show'); }
  document.addEventListener('pointerdown', (e) => { if (menu.classList.contains('show') && !menu.contains(e.target)) closeMenu(); }, true);

  const modal = portal('gh-modal', '<div class="gh-dialog"></div>');
  let modalDone = null;
  function confirmDialog({ title, text, list = [], ok = 'OK', danger = false }) {
    return new Promise((resolve) => {
      modal.firstElementChild.innerHTML = `<h3>${esc(title)}</h3><p>${esc(text)}</p>` + (list.length ? `<ul>${list.slice(0, 40).map((x) => `<li>${esc(x)}</li>`).join('')}${list.length > 40 ? `<li>… and ${list.length - 40} more</li>` : ''}</ul>` : '') +
        `<div class="btns"><button class="gh-btn" data-r="0">Cancel</button><button class="gh-btn ${danger ? 'dangerfill' : 'primary'}" data-r="1">${esc(ok)}</button></div>`;
      modal.classList.add('show');
      modalDone = (v) => { modal.classList.remove('show'); modalDone = null; resolve(v); };
      modal.querySelector('[data-r="1"]').focus();
    });
  }
  modal.addEventListener('click', (e) => { const b = e.target.closest('[data-r]'); if (b && modalDone) modalDone(b.dataset.r === '1'); else if (e.target === modal && modalDone) modalDone(false); });

  function toggleFullscreen(el) {
    try {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (el.requestFullscreen) el.requestFullscreen().catch(() => toast('Full screen is not available here.', 'bad'));
    } catch (e) { /* not supported */ }
  }
  function showShortcuts() {
    const rows = [
      ['Gallery', [['/', 'Search'], ['← → ↑ ↓', 'Move between cards'], ['Enter', 'Open'], ['Space', 'Select'], ['Shift+click', 'Select a range'],
        ['Ctrl+click', 'Add to the selection'], ['Ctrl+A', 'Select everything shown'], ['F', 'Favorite'], ['Delete', 'Delete'], ['Esc', 'Clear the selection']]],
      ['Viewer', [['← →', 'Previous / next'], ['O · W · S', 'Output · input vs output · side by side'], ['Wheel', 'Zoom at the cursor'],
        ['0 · 1 · 2', 'Fit · 100 % · 200 %'], ['+ −', 'Zoom in / out'], ['[ ]', 'Move the slider'], ['Space', 'Play / pause'], [', .', 'Previous / next frame'],
        ['M', 'Sound'], ['R', 'Reload parameters'], ['C', 'Compare with…'], ['D', 'Download'], ['I', 'Details panel'], ['Esc', 'Close']]],
      ['Compare', [['X', 'Swap A and B'], ['W · S', 'Slider · side by side'], ['I', 'Differences'], ['Space , .', 'Play and step together']]],
    ];
    modal.firstElementChild.innerHTML = '<h3>⌨ Keyboard shortcuts</h3>' + rows.map(([t, list]) => `<p style="margin:12px 0 6px;color:#e6ebf4;font-weight:700">${t}</p>` +
      `<div style="display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px">${list.map(([k, v]) => `<span style="font:600 12px/1.6 'JetBrains Mono',Consolas,monospace;color:#c7d2fe;white-space:nowrap">${esc(k)}</span><span style="color:#a8b3c7">${esc(v)}</span>`).join('')}</div>`).join('') +
      '<div class="btns" style="margin-top:16px"><button class="gh-btn primary" data-r="1">Got it</button></div>';
    modal.classList.add('show');
    modalDone = () => { modal.classList.remove('show'); modalDone = null; };
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); } catch (e) {
      const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch (e2) { /* ignore */ } ta.remove();
    }
    toast('📋 Copied: ' + (text.length > 90 ? '…' + text.slice(-88) : text), 'ok', 2200);
  }
  function download(url, name) {
    const a = document.createElement('a'); a.href = U(url); a.download = name || ''; a.rel = 'noopener';
    document.body.appendChild(a); a.click(); a.remove();
  }

  // ------------------------------------------------------------------ state
  const S = {
    items: [], byId: new Map(), engines: {}, sendTabs: [], videoOnly: [], filtered: [], counts: {},
    q: '', kind: LS.get('kind', 'all'), when: LS.get('when', 'all'), status: LS.get('status', 'done'),
    sort: LS.get('sort', 'new'), view: LS.get('view', 'grid'), size: LS.get('size', 236), fav: false,
    engineSel: new Set(LS.get('engines', [])), sel: new Set(), last: null, sig: '', root: '', total: 0,
    loading: false, loaded: false, rendered: 0, pickFor: null, hoverCard: null, hoverVideo: null, hoverTimer: 0,
    itemsSig: '',
  };
  const CHUNK = 72;

  // ------------------------------------------------------------------ layout
  const root = document.createElement('div');
  root.className = 'gh-root';
  root.tabIndex = -1;
  host.appendChild(root);
  root.innerHTML = `
  <div class="gh-head">
    <div class="gh-head-title"><div class="gh-logo">🗂️</div>
      <div style="min-width:0"><div class="gh-h1">Past Generations</div><div class="gh-stats">Loading your generations…</div></div></div>
    <div class="gh-head-actions">
      <button class="gh-btn" data-a="keys" title="Keyboard shortcuts (?)"><span class="gh-ico">⌨</span>Shortcuts</button>
      <button class="gh-btn" data-a="refresh" title="Scan the outputs folder again"><span class="gh-ico">⟳</span>Refresh</button>
      <button class="gh-btn" data-a="openroot" title="Open the outputs folder on the computer running the app"><span class="gh-ico">📂</span>Outputs folder</button>
    </div>
  </div>
  <div class="gh-bar">
    <div class="gh-row">
      <label class="gh-search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        <input type="search" placeholder="Search file name, engine, model, prompt, 3840, 4x…" spellcheck="false" autocomplete="off">
        <button class="gh-clear-q" title="Clear search">✕</button><span class="gh-kbd">/</span></label>
      <div class="gh-seg" data-k="kind"><button data-v="all">All</button><button data-v="video">🎬 Videos</button><button data-v="image">🖼️ Images</button></div>
      <div class="gh-seg" data-k="when"><button data-v="all">Any time</button><button data-v="today">Today</button><button data-v="week">7 days</button><button data-v="month">30 days</button></div>
      <button class="gh-btn" data-a="fav" title="Show favorites only"><span class="gh-ico">★</span>Favorites</button>
    </div>
    <div class="gh-row">
      <div class="gh-chips"></div>
      <div style="flex:1"></div>
      <select class="gh-select" data-k="status" title="Which runs to show">
        <option value="done">✅ Finished</option><option value="problems">⚠️ Failed or cancelled</option><option value="all">All runs</option></select>
      <select class="gh-select" data-k="sort" title="Sort order">
        <option value="new">Newest first</option><option value="old">Oldest first</option><option value="res">Highest resolution</option>
        <option value="size">Largest file</option><option value="dur">Longest video</option><option value="name">Name A–Z</option><option value="engine">Engine</option></select>
      <span class="gh-size" title="Card size">▫<input type="range" min="160" max="420" step="4">◻</span>
      <div class="gh-seg" data-k="view"><button data-v="grid" title="Grid">▦</button><button data-v="list" title="List">☰</button></div>
    </div>
    <div class="gh-selbar">
      <span class="count"></span>
      <button class="gh-btn primary" data-a="compare"><span class="gh-ico">⚖️</span>Compare</button>
      <button class="gh-btn" data-a="selfav"><span class="gh-ico">★</span>Favorite</button>
      <button class="gh-btn" data-a="seldl"><span class="gh-ico">⬇</span>Download</button>
      <button class="gh-btn danger" data-a="seldel"><span class="gh-ico">🗑</span>Delete</button>
      <button class="gh-btn" data-a="selall">Select all</button>
      <button class="gh-btn" data-a="selnone">Clear</button>
      <span class="hint">Shift+click selects a range · Ctrl+click adds one</span>
    </div>
  </div>
  <div class="gh-content"></div>
  <div class="gh-empty"></div>`;
  const $ = (s) => root.querySelector(s);
  const content = $('.gh-content');
  const emptyBox = $('.gh-empty');
  const searchInput = $('.gh-search input');
  const sentinel = document.createElement('div');
  sentinel.className = 'gh-sentinel';

  function syncControls() {
    root.querySelectorAll('.gh-seg[data-k]').forEach((seg) => seg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', S[seg.dataset.k] === b.dataset.v)));
    root.querySelector('select[data-k="status"]').value = S.status;
    root.querySelector('select[data-k="sort"]').value = S.sort;
    root.querySelector('.gh-size input').value = S.size;
    root.style.setProperty('--gh-size', S.size + 'px');
    root.classList.toggle('list', S.view === 'list');
    root.querySelector('[data-a="fav"]').classList.toggle('on', S.fav);
    $('.gh-search').classList.toggle('has-q', !!S.q);
  }
  syncControls();

  // skeleton while the first index loads
  content.innerHTML = '<div class="gh-grid">' + Array.from({ length: 10 }, () => '<div class="gh-card" style="--c:#334155"><div class="gh-media"><div class="gh-skel"></div></div><div class="gh-info"><div class="gh-name">&nbsp;</div><div class="gh-sub"><span>&nbsp;</span></div></div></div>').join('') + '</div>';

  // ------------------------------------------------------------------ data
  function setItems(res) {
    S.engines = res.engines || S.engines;
    S.sendTabs = res.send_tabs || S.sendTabs;
    S.videoOnly = res.video_only || S.videoOnly;
    S.root = res.root || '';
    S.sig = res.sig || '';
    S.total = res.total_size || 0;
    const prevThumbs = new Map(S.items.map((i) => [i.id, i.thumb]));
    S.items = (res.items || []).map((it) => {
      const res2 = fmtRes(it.w, it.h);
      it._s = `${it.search} ${res2.replace('×', 'x')} ${it.w || ''} ${it.h || ''} ${it.scale ? it.scale + 'x' : ''} ${it.kind} ${statusName[it.status] || ''}`.toLowerCase();
      if (!it.thumb && prevThumbs.get(it.id)) it.thumb = prevThumbs.get(it.id);
      return it;
    });
    S.byId = new Map(S.items.map((i) => [i.id, i]));
    for (const id of [...S.sel]) if (!S.byId.has(id)) S.sel.delete(id);
  }
  const itemsSig = (items) => items.map((i) => `${i.id}:${i.ts}:${i.w || 0}:${i.fav ? 1 : 0}:${i.status}:${i.thumb ? 1 : 0}`).join('|');

  async function refresh({ force = false, silent = false } = {}) {
    if (S.loading) return;
    S.loading = true;
    const btn = root.querySelector('[data-a="refresh"]');
    btn.classList.add('gh-spin');
    const res = await call('hist_index', { force });
    btn.classList.remove('gh-spin');
    S.loading = false;
    if (!res) return;
    const prev = new Set(S.items.map((i) => i.id));
    const first = !S.loaded;
    setItems(res);
    S.loaded = true;
    const sig = itemsSig(S.items);
    const fresh = first ? [] : S.items.filter((i) => !prev.has(i.id)).map((i) => i.id);
    if (sig !== S.itemsSig || first || force) {
      S.itemsSig = sig;
      render(true);
      fresh.forEach((id) => content.querySelectorAll(`.gh-card[data-id="${id}"]`).forEach((c) => c.classList.add('gh-new')));
    }
    if (fresh.length && silent) toast(`✨ ${fresh.length} new generation${fresh.length > 1 ? 's' : ''}`, 'ok');
    if (!silent && force) toast(`⟳ ${S.items.length} generations indexed`, 'ok', 1800);
    if (res.pending > 0) setTimeout(() => refresh({ silent: true }), 1400);
  }

  // ------------------------------------------------------------------ filtering / sorting
  function applyFilters() {
    const terms = S.q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const now = Date.now() / 1000;
    const limit = { today: dayStart(), week: now - 7 * 86400, month: now - 30 * 86400 }[S.when] || 0;
    const counts = {};
    const base = S.items.filter((it) => {
      if (S.kind !== 'all' && it.kind !== S.kind) return false;
      if (S.status === 'done' && it.status !== 'ok') return false;
      if (S.status === 'problems' && it.status === 'ok') return false;
      if (S.fav && !it.fav) return false;
      if (limit && (it.ts || 0) < limit) return false;
      if (terms.length && !terms.every((t) => it._s.includes(t))) return false;
      return true;
    });
    base.forEach((it) => { counts[it.engine] = (counts[it.engine] || 0) + 1; });
    S.counts = counts;
    const list = S.engineSel.size ? base.filter((it) => S.engineSel.has(it.engine)) : base;
    const by = {
      new: (a, b) => (b.ts || 0) - (a.ts || 0),
      old: (a, b) => (a.ts || 0) - (b.ts || 0),
      res: (a, b) => ((b.w || 0) * (b.h || 0)) - ((a.w || 0) * (a.h || 0)) || (b.ts - a.ts),
      size: (a, b) => (b.size || 0) - (a.size || 0),
      dur: (a, b) => (b.dur || 0) - (a.dur || 0) || (b.ts - a.ts),
      name: (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }),
      engine: (a, b) => a.label.localeCompare(b.label) || (b.ts - a.ts),
    }[S.sort] || ((a, b) => b.ts - a.ts);
    list.sort(by);
    S.filtered = list;
  }

  function renderChips() {
    const box = $('.gh-chips');
    const present = Object.keys(S.engines).filter((k) => S.counts[k] || S.engineSel.has(k));
    const total = Object.values(S.counts).reduce((a, b) => a + b, 0);
    box.innerHTML = `<button class="gh-chip ${S.engineSel.size ? '' : 'on'}" data-eng="*" style="--c:#818cf8"><span class="dot"></span>All engines <span class="n">${total}</span></button>` +
      present.map((k) => { const e = S.engines[k]; return `<button class="gh-chip ${S.engineSel.has(k) ? 'on' : ''}" data-eng="${k}" style="--c:${e.color}"><span class="dot"></span>${esc(e.icon)} ${esc(e.label)} <span class="n">${S.counts[k] || 0}</span></button>`; }).join('');
  }

  function renderStats() {
    const vids = S.items.filter((i) => i.kind === 'video').length;
    const imgs = S.items.filter((i) => i.kind === 'image').length;
    const shown = S.filtered.length;
    const el = $('.gh-stats');
    el.innerHTML = `<b>${S.items.length}</b> generations · <b>${vids}</b> videos · <b>${imgs}</b> images · <b>${fmtSize(S.total) || '0 B'}</b>` +
      (shown !== S.items.length ? ` · showing <b>${shown}</b>` : '');
    el.title = S.root ? `Outputs folder: ${S.root}` : '';
  }

  // ------------------------------------------------------------------ rendering
  const grouping = () => S.sort === 'new' || S.sort === 'old';
  let curGrid = null; let curGroup = null; let groupCounts = {};

  function render(keepCount = false) {
    stopHover();
    applyFilters();
    renderChips();
    renderStats();
    updateSelbar();
    const want = keepCount ? Math.max(CHUNK, S.rendered) : CHUNK;
    content.innerHTML = '';
    S.rendered = 0; curGrid = null; curGroup = null;
    groupCounts = {};
    if (grouping()) S.filtered.forEach((it) => { const k = bucketOf(it.ts || 0)[0]; groupCounts[k] = (groupCounts[k] || 0) + 1; });
    if (!S.filtered.length) {
      emptyBox.classList.add('show');
      if (!S.items.length) {
        emptyBox.innerHTML = `<div class="big">🌌</div><h3>No generations yet</h3><p>Everything you upscale, restore or chain appears here automatically, with its settings, its input and a preview.</p><p style="font-size:12px">Outputs folder: <code>${esc(S.root)}</code></p>`;
      } else {
        emptyBox.innerHTML = `<div class="big">🔎</div><h3>Nothing matches these filters</h3><p>${S.items.length} generations are hidden by the search or filters.</p><button class="gh-btn primary" data-a="clearfilters"><span class="gh-ico">✕</span>Clear all filters</button>`;
      }
      return;
    }
    emptyBox.classList.remove('show');
    renderMore(want);
  }

  function renderMore(count = CHUNK) {
    const end = Math.min(S.filtered.length, S.rendered + count);
    for (let i = S.rendered; i < end; i++) {
      const it = S.filtered[i];
      if (!curGrid || (grouping() && bucketOf(it.ts || 0)[0] !== curGroup)) {
        const sec = document.createElement('section');
        sec.className = 'gh-group';
        if (grouping()) {
          const [key, label] = bucketOf(it.ts || 0);
          curGroup = key;
          sec.innerHTML = `<div class="gh-ghead"><h3>${esc(label)}</h3><span>${groupCounts[key] || 0} item${groupCounts[key] === 1 ? '' : 's'}</span></div>`;
        }
        curGrid = document.createElement('div');
        curGrid.className = 'gh-grid';
        sec.appendChild(curGrid);
        content.appendChild(sec);
      }
      curGrid.appendChild(buildCard(it));
    }
    S.rendered = end;
    content.appendChild(sentinel);
    if (S.rendered < S.filtered.length) moreIO.observe(sentinel); else moreIO.unobserve(sentinel);
  }

  function cardHTML(it) {
    const res = fmtRes(it.w, it.h);
    const scale = it.scale && it.scale !== 1 ? `<span class="x"> · ${it.scale}×</span>` : '';
    const dur = it.kind === 'video' ? `▶ ${fmtDur(it.dur) || 'video'}` : (it.kind === 'image' ? '🖼️ image' : '—');
    const st = it.status !== 'ok' ? `<span class="gh-badge gh-st ${it.status}">${esc(statusName[it.status] || it.status)}</span>` : '';
    const sub = it.chain_run ? `🔗 ${it.model || it.chain}` : (it.model || it.label);
    const thumb = it.thumb ? U(it.thumb) : '';
    const noPrev = !it.out ? `<div class="gh-noprev"><div><b>${it.status === 'ok' ? '📄' : '⛔'}</b>${esc(statusName[it.status] || 'No output')}</div></div>` : '';
    return `<div class="gh-media${!it.out ? ' loaded' : ''}">
        <div class="gh-skel"></div>${noPrev}
        ${thumb ? `<img class="bg" src="${esc(thumb)}" alt="" loading="lazy" decoding="async"><img class="fg" src="${esc(thumb)}" alt="" loading="lazy" decoding="async">` : ''}
        <span class="gh-badge gh-eng">${esc(it.icon)} ${esc(it.label)}</span>${st}
        <div class="gh-tools"><button class="gh-tool star${it.fav ? ' on' : ''}" title="Favorite (F)">${it.fav ? '★' : '☆'}</button><button class="gh-tool chk${S.sel.has(it.id) ? ' on' : ''}" title="Select (Ctrl+click)">✓</button></div>
        <span class="gh-badge gh-dur">${dur}</span>${res ? `<span class="gh-badge gh-res">${res}${scale}</span>` : ''}
      </div>
      <div class="gh-info">
        <div class="gh-name" title="${esc(it.out || it.dir)}">${esc(it.name || '(no output)')}</div>
        <div class="gh-sub"><span title="${esc(sub)}">${esc(sub)}</span><span title="${esc(fullDate(it.ts))}">${esc(relTime(it.ts))}</span></div>
        <div class="gh-col"><span class="gh-pill" style="--c:${it.color}"><i></i>${esc(it.label)}</span></div>
        <div class="gh-col">${esc(it.model || '')}</div>
        <div class="gh-col m">${res}${it.scale && it.scale !== 1 ? ' · ' + it.scale + '×' : ''}</div>
        <div class="gh-col m">${it.kind === 'video' ? fmtDur(it.dur) : 'image'}</div>
        <div class="gh-col m">${fmtSize(it.size)}</div>
        <div class="gh-col" title="${esc(fullDate(it.ts))}">${esc(relTime(it.ts))}</div>
        <div class="gh-col">${it.status === 'ok' ? '✅' : '⚠️'} ${esc(statusName[it.status] || '')}</div>
        <div class="gh-lt"><button class="gh-tool star${it.fav ? ' on' : ''}" style="opacity:1;transform:none;position:static">${it.fav ? '★' : '☆'}</button><button class="gh-tool chk${S.sel.has(it.id) ? ' on' : ''}" style="opacity:1;transform:none;position:static">✓</button></div>
      </div>`;
  }

  function buildCard(it) {
    const card = document.createElement('div');
    card.className = 'gh-card' + (S.sel.has(it.id) ? ' sel' : '');
    card.dataset.id = it.id;
    card.tabIndex = 0;
    card.style.setProperty('--c', it.color);
    card.innerHTML = cardHTML(it);
    wireThumb(card, it);
    return card;
  }

  function wireThumb(card, it) {
    const media = card.querySelector('.gh-media');
    const fg = media.querySelector('img.fg');
    if (fg) {
      const done = () => media.classList.add('loaded');
      if (fg.complete && fg.naturalWidth) done(); else { fg.addEventListener('load', done, { once: true }); fg.addEventListener('error', () => { media.classList.add('loaded'); }, { once: true }); }
    } else if (it.out) {
      thumbIO.observe(card);
    }
  }

  function updateCard(id) {
    const it = S.byId.get(id);
    if (!it) return;
    content.querySelectorAll(`.gh-card[data-id="${id}"]`).forEach((card) => {
      card.classList.toggle('sel', S.sel.has(id));
      card.innerHTML = cardHTML(it);
      wireThumb(card, it);
      if (!it.thumb && it.thumbFailed) {
        const m = card.querySelector('.gh-media');
        m.classList.add('loaded');
        if (!m.querySelector('.gh-noprev')) m.insertAdjacentHTML('afterbegin', `<div class="gh-noprev"><div><b>${it.kind === 'video' ? '🎬' : '🖼️'}</b>No preview</div></div>`);
      }
    });
  }

  // thumbnails: requested in small batches as cards come near the viewport
  const thumbQ = new Set(); let thumbBusy = 0;
  const thumbIO = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      thumbIO.unobserve(e.target);
      const it = S.byId.get(e.target.dataset.id);
      if (it && !it.thumb && !it.thumbFailed && it.out) thumbQ.add(it.id);
    }
    pumpThumbs();
  }, { rootMargin: '700px 0px' });
  function pumpThumbs() {
    while (thumbBusy < 2 && thumbQ.size) {
      const ids = [...thumbQ].slice(0, 10);
      ids.forEach((i) => thumbQ.delete(i));
      thumbBusy++;
      call('hist_thumbs', { ids }).then((res) => {
        for (const id of ids) {
          const it = S.byId.get(id);
          if (!it) continue;
          const url = res && res[id];
          if (url) it.thumb = url; else it.thumbFailed = true;
          updateCard(id);
        }
      }).finally(() => { thumbBusy--; pumpThumbs(); });
    }
  }
  const moreIO = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) renderMore(); }, { rootMargin: '1400px 0px' });

  // hover preview of videos
  function stopHover() {
    clearTimeout(S.hoverTimer);
    if (S.hoverVideo) { const v = S.hoverVideo; try { v.pause(); v.removeAttribute('src'); v.load(); } catch (e) { /* ignore */ } v.remove(); S.hoverVideo = null; }
    S.hoverCard = null;
  }
  content.addEventListener('pointerover', (e) => {
    const card = e.target.closest('.gh-card');
    if (!card || card === S.hoverCard) return;
    stopHover();
    S.hoverCard = card;
    const it = S.byId.get(card.dataset.id);
    if (!it || it.kind !== 'video' || !it.playable || S.view === 'list' || !it.url) return;
    S.hoverTimer = setTimeout(() => {
      if (S.hoverCard !== card) return;
      const v = document.createElement('video');
      v.className = 'hover'; v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto';
      v.src = U(it.url);
      v.addEventListener('playing', () => v.classList.add('live'), { once: true });
      card.querySelector('.gh-media').appendChild(v);
      S.hoverVideo = v;
      v.play().catch(() => {});
    }, 260);
  });
  content.addEventListener('pointerout', (e) => {
    const card = e.target.closest('.gh-card');
    if (card && card === S.hoverCard && !card.contains(e.relatedTarget)) stopHover();
  });

  // ------------------------------------------------------------------ selection
  function updateSelbar() {
    const bar = $('.gh-selbar');
    const n = S.sel.size;
    bar.classList.toggle('show', n > 0);
    root.classList.toggle('selecting', n > 0);
    bar.querySelector('.count').textContent = `${n} selected`;
    const cmp = bar.querySelector('[data-a="compare"]');
    cmp.disabled = n !== 2;
    cmp.title = n === 2 ? 'Compare the two selected generations' : 'Select exactly two generations to compare';
  }
  function toggleSel(id, on) {
    const want = on === undefined ? !S.sel.has(id) : on;
    if (want) S.sel.add(id); else S.sel.delete(id);
    S.last = id;
    content.querySelectorAll(`.gh-card[data-id="${id}"]`).forEach((c) => { c.classList.toggle('sel', want); c.querySelectorAll('.gh-tool.chk').forEach((b) => b.classList.toggle('on', want)); });
    updateSelbar();
  }
  function rangeSel(fromId, toId) {
    const ids = S.filtered.map((i) => i.id);
    let a = ids.indexOf(fromId); let b = ids.indexOf(toId);
    if (a < 0 || b < 0) { toggleSel(toId); return; }
    if (a > b) [a, b] = [b, a];
    ids.slice(a, b + 1).forEach((id) => toggleSel(id, true));
    S.last = toId;
  }
  function clearSel() { [...S.sel].forEach((id) => toggleSel(id, false)); }

  // ------------------------------------------------------------------ actions shared by grid, viewer and compare
  async function toggleFav(ids, value) {
    const items = ids.map((id) => S.byId.get(id)).filter(Boolean);
    if (!items.length) return;
    const want = value === undefined ? !items.every((i) => i.fav) : value;
    const res = await call('hist_favorite', { ids, value: want });
    if (res == null) return;
    items.forEach((i) => { i.fav = want; updateCard(i.id); });
    S.itemsSig = itemsSig(S.items);
    if (V.open && V.it && ids.includes(V.it.id)) viewerFavState();
    if (S.fav && !want) render(true);
    toast(want ? `★ Added ${items.length > 1 ? items.length + ' generations' : 'to favorites'}` : '☆ Removed from favorites', 'ok', 1600);
  }
  async function downloadItems(ids) {
    const items = ids.map((id) => S.byId.get(id)).filter((i) => i && i.url);
    if (!items.length) { toast('Nothing to download.', 'bad'); return; }
    if (items.length === 1) { download(items[0].url, items[0].name); return; }
    toast(`📦 Packing ${items.length} files…`, '', 2500);
    const res = await call('hist_zip', { ids: items.map((i) => i.id) });
    if (res && res.url) download(res.url, res.name); else toast((res && res.msg) || 'Could not build the zip.', 'bad');
  }
  async function deleteItems(ids) {
    const items = ids.map((id) => S.byId.get(id)).filter(Boolean);
    if (!items.length) return false;
    const ok = await confirmDialog({
      title: items.length === 1 ? 'Delete this generation?' : `Delete ${items.length} generations?`,
      text: 'Each run folder (or file) moves to the Recycle Bin, so you can restore it there.',
      list: items.map((i) => `${i.name || i.dir}  ·  ${i.label}  ·  ${relTime(i.ts)}`), ok: 'Move to Recycle Bin', danger: true,
    });
    if (!ok) return false;
    const res = await call('hist_delete', { ids: items.map((i) => i.id) });
    if (!res) return false;
    const gone = new Set(res.removed || []);
    S.items = S.items.filter((i) => !gone.has(i.id));
    S.byId = new Map(S.items.map((i) => [i.id, i]));
    gone.forEach((id) => S.sel.delete(id));
    S.itemsSig = itemsSig(S.items);
    render(true);
    if (gone.size) toast(`🗑 Moved ${gone.size} to ${res.where}`, 'ok');
    if ((res.failed || []).length) toast(`⚠️ ${res.failed.length} could not be moved (in use?)`, 'bad', 4500);
    return gone.size > 0;
  }
  function sendAction(action, it, tab) {
    trigger('select', { index: 0, value: { action, id: it.id, tab: tab || '' } });
    const target = tab || (it.chain_run ? 'chains' : it.tab);
    const eng = Object.values(S.engines).find((e) => e.tab === target);
    toast(action === 'send' ? `➜ Opening ${eng ? eng.label : target} with this file as input…` : `↻ Loading the settings of this run into ${eng ? eng.label : target}…`, '', 2600);
    closeViewer(); closeCompare();
  }
  function sendMenu(anchor, it) {
    const entries = S.sendTabs.filter((t) => !(it.kind === 'image' && S.videoOnly.includes(t))).map((t) => {
      const e = Object.values(S.engines).find((x) => x.tab === t) || { label: t, icon: '', color: '#64748b' };
      return { label: t === 'chains' ? 'Chains (as chain input)' : e.label, icon: e.icon, color: e.color, run: () => sendAction('send', it, t) };
    });
    openMenu(anchor, 'Use this output as the input of', entries);
  }
  function startPick(id) {
    const it = S.byId.get(id);
    if (!it) return;
    S.pickFor = id;
    closeViewer();
    pickHint.querySelector('span').textContent = `Pick a second generation to compare with “${it.name}”`;
    pickHint.classList.add('show');
  }
  function endPick() { S.pickFor = null; pickHint.classList.remove('show'); }
  const pickHint = portal('gh-pickhint', '<span></span><button>Cancel</button>');
  pickHint.querySelector('button').onclick = endPick;

  // ------------------------------------------------------------------ toolbar events
  let qTimer = 0;
  searchInput.addEventListener('input', () => {
    clearTimeout(qTimer);
    qTimer = setTimeout(() => { S.q = searchInput.value; $('.gh-search').classList.toggle('has-q', !!S.q); render(); }, 110);
  });
  $('.gh-clear-q').addEventListener('click', (e) => { e.preventDefault(); searchInput.value = ''; S.q = ''; syncControls(); render(); searchInput.focus(); });
  root.querySelectorAll('.gh-seg[data-k]').forEach((seg) => seg.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-v]');
    if (!b) return;
    S[seg.dataset.k] = b.dataset.v;
    LS.set(seg.dataset.k, b.dataset.v);
    syncControls();
    render();
  }));
  root.querySelectorAll('select[data-k]').forEach((sel) => sel.addEventListener('change', () => { S[sel.dataset.k] = sel.value; LS.set(sel.dataset.k, sel.value); render(); }));
  $('.gh-size input').addEventListener('input', (e) => { S.size = +e.target.value; root.style.setProperty('--gh-size', S.size + 'px'); LS.set('size', S.size); });
  $('.gh-chips').addEventListener('click', (e) => {
    const c = e.target.closest('[data-eng]');
    if (!c) return;
    const k = c.dataset.eng;
    if (k === '*') S.engineSel.clear();
    else if (e.ctrlKey || e.metaKey || e.shiftKey) { if (S.engineSel.has(k)) S.engineSel.delete(k); else S.engineSel.add(k); } else if (S.engineSel.size === 1 && S.engineSel.has(k)) S.engineSel.clear();
    else { S.engineSel.clear(); S.engineSel.add(k); }
    LS.set('engines', [...S.engineSel]);
    render();
  });
  root.addEventListener('click', async (e) => {
    const a = e.target.closest('[data-a]');
    if (!a || !root.contains(a)) return;
    const act = a.dataset.a;
    if (act === 'refresh') refresh({ force: true });
    else if (act === 'openroot') { const r = await call('hist_open', { id: '__root__', what: 'folder' }); if (r) toast(r.msg, r.ok ? 'ok' : 'bad'); }
    else if (act === 'fav') { S.fav = !S.fav; syncControls(); render(); }
    else if (act === 'keys') showShortcuts();
    else if (act === 'clearfilters') { S.q = ''; searchInput.value = ''; S.kind = 'all'; S.when = 'all'; S.status = 'all'; S.fav = false; S.engineSel.clear(); ['kind', 'when', 'status'].forEach((k) => LS.set(k, S[k])); LS.set('engines', []); syncControls(); render(); }
    else if (act === 'compare') { const ids = [...S.sel]; if (ids.length === 2) { const [x, y] = ids.map((id) => S.byId.get(id)).sort((p, q) => (p.ts || 0) - (q.ts || 0)); openCompare(x.id, y.id); C.fromSel = true; } }
    else if (act === 'selfav') toggleFav([...S.sel]);
    else if (act === 'seldl') downloadItems([...S.sel]);
    else if (act === 'seldel') deleteItems([...S.sel]);
    else if (act === 'selall') { S.filtered.forEach((i) => S.sel.add(i.id)); render(true); }
    else if (act === 'selnone') clearSel();
  });

  content.addEventListener('click', (e) => {
    const card = e.target.closest('.gh-card');
    if (!card || !S.loaded) return;
    const id = card.dataset.id;
    if (e.target.closest('.gh-tool.star')) { e.stopPropagation(); toggleFav([id]); return; }
    if (e.target.closest('.gh-tool.chk')) { e.stopPropagation(); if (e.shiftKey && S.last) rangeSel(S.last, id); else toggleSel(id); return; }
    if (S.pickFor) { const a = S.pickFor; endPick(); if (a !== id) openCompare(a, id); return; }
    if (e.shiftKey && S.last) { rangeSel(S.last, id); return; }
    if (e.ctrlKey || e.metaKey || S.sel.size) { toggleSel(id); return; }
    openViewer(id);
  });
  content.addEventListener('dblclick', (e) => { const card = e.target.closest('.gh-card'); if (card && S.sel.size) { openViewer(card.dataset.id); } });
  content.addEventListener('keydown', (e) => {
    const card = e.target.closest && e.target.closest('.gh-card');
    if (!card) return;
    if (e.key === 'Enter') { e.preventDefault(); openViewer(card.dataset.id); }
    else if (e.key === ' ') { e.preventDefault(); toggleSel(card.dataset.id); }
    else if (e.key === 'f' || e.key === 'F') toggleFav([card.dataset.id]);
    else if (e.key === 'Delete') deleteItems([card.dataset.id]);
    else if (['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(e.key)) {
      const cards = [...content.querySelectorAll('.gh-card')];
      const i = cards.indexOf(card);
      const perRow = Math.max(1, Math.round(card.parentElement.clientWidth / (card.offsetWidth + 16)));
      const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: perRow, ArrowUp: -perRow }[e.key];
      const next = cards[Math.min(cards.length - 1, Math.max(0, i + step))];
      if (next) { e.preventDefault(); next.focus(); next.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    }
  });

  // ================================================================== STAGE: zoom / pan / wipe / side-by-side / synced video
  class Stage {
    constructor(wrap) {
      this.wrap = wrap;
      wrap.innerHTML = `<div class="gh-stage"><div class="gh-zoomtag"></div><div class="gh-loading"><div><div class="ring"></div><div class="msg">Loading…</div></div></div></div>
        <div class="gh-transport">
          <button class="gh-btn icon" data-t="back" title="Previous frame (,)">⏮</button>
          <button class="gh-btn icon" data-t="play" title="Play / pause (Space)">▶</button>
          <button class="gh-btn icon" data-t="fwd" title="Next frame (.)">⏭</button>
          <span class="time">0:00.00 / 0:00.00</span>
          <input class="gh-scrub" type="range" min="0" max="1000" step="1" value="0">
          <select class="gh-select" data-t="rate" title="Speed" style="height:32px"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
          <button class="gh-btn icon on" data-t="loop" title="Loop">🔁</button>
          <button class="gh-btn icon" data-t="mute" title="Sound on / off (M)">🔊</button>
        </div>
        <div class="gh-zoombar">
          <button class="gh-btn" data-z="fit" title="Fit (0)">Fit</button><button class="gh-btn" data-z="one" title="Actual pixels (1)">100%</button>
          <button class="gh-btn" data-z="two" title="200% (2)">200%</button><button class="gh-btn icon" data-z="out" title="Zoom out (−)">−</button><button class="gh-btn icon" data-z="in" title="Zoom in (+)">+</button>
          <span class="hint">Scroll to zoom at the cursor · drag to pan · double-click for 100%</span>
        </div>`;
      this.st = wrap.querySelector('.gh-stage');
      this.tag = wrap.querySelector('.gh-zoomtag');
      this.loadingEl = wrap.querySelector('.gh-loading');
      this.tp = wrap.querySelector('.gh-transport');
      this.scrub = wrap.querySelector('.gh-scrub');
      this.timeEl = wrap.querySelector('.gh-transport .time');
      this.mode = 'single'; this.items = []; this.panes = []; this.z = 1; this.fx = 0; this.fy = 0; this.split = 0.5;
      this.muted = LS.get('muted', false); this.loop = true; this.raf = 0; this.token = 0;
      this.bind();
      new ResizeObserver(() => this.layout()).observe(this.st);
    }

    bind() {
      const st = this.st;
      st.addEventListener('wheel', (e) => {
        if (!this.panes.length) return;
        e.preventDefault();
        const p = this.paneAt(e.clientX);
        const f = Math.exp(-(e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY) * 0.0016);
        this.zoomAt(this.z * f, e.clientX, e.clientY, p);
      }, { passive: false });
      let drag = null;
      st.addEventListener('pointerdown', (e) => {
        if (e.button !== 0 || !this.panes.length) return;
        const onHandle = e.target.closest('.gh-handle');
        if (this.mode === 'wipe' && (onHandle || this.z <= 1.001)) {
          drag = { kind: 'split' }; this.setSplitFromX(e.clientX);
        } else if (this.z > 1.001) {
          drag = { kind: 'pan', x: e.clientX, y: e.clientY, fx: this.fx, fy: this.fy }; st.classList.add('panning');
        } else return;
        st.setPointerCapture(e.pointerId);
        e.preventDefault();
      });
      st.addEventListener('pointermove', (e) => {
        if (!drag) return;
        if (drag.kind === 'split') this.setSplitFromX(e.clientX);
        else {
          const f = this.panes[0].fit;
          this.fx = drag.fx + (e.clientX - drag.x) / f.w;
          this.fy = drag.fy + (e.clientY - drag.y) / f.h;
          this.clamp(); this.apply();
        }
      });
      const end = () => { drag = null; st.classList.remove('panning'); };
      st.addEventListener('pointerup', end); st.addEventListener('pointercancel', end);
      st.addEventListener('dblclick', (e) => {
        if (!this.panes.length || e.target.closest('.gh-handle')) return;
        if (this.z > 1.01) this.reset(); else this.zoomAt(this.oneToOne(), e.clientX, e.clientY, this.paneAt(e.clientX));
      });
      this.wrap.querySelector('.gh-zoombar').addEventListener('click', (e) => {
        const b = e.target.closest('[data-z]'); if (!b) return;
        this.zoomCmd(b.dataset.z);
      });
      this.tp.addEventListener('click', (e) => {
        const b = e.target.closest('[data-t]'); if (!b) return;
        const t = b.dataset.t;
        if (t === 'play') this.toggle();
        else if (t === 'back') this.step(-1);
        else if (t === 'fwd') this.step(1);
        else if (t === 'loop') { this.loop = !this.loop; b.classList.toggle('on', this.loop); this.videos().forEach((v) => { v.loop = this.loop; }); }
        else if (t === 'mute') this.toggleMute();
      });
      this.tp.querySelector('[data-t="rate"]').addEventListener('change', (e) => this.videos().forEach((v) => { v.playbackRate = +e.target.value; }));
      this.scrub.addEventListener('input', () => {
        const m = this.master(); if (!m || !m.duration) return;
        const t = (this.scrub.value / 1000) * m.duration;
        this.videos().forEach((v) => { try { v.currentTime = Math.min(t, (v.duration || t) - 0.001); } catch (err) { /* not ready */ } });
        this.updTime();
      });
    }

    zoomCmd(cmd) {
      const r = this.st.getBoundingClientRect();
      const cx = r.left + r.width / 2; const cy = r.top + r.height / 2;
      const p = this.panes[0];
      if (!p) return;
      if (cmd === 'fit') this.reset();
      else if (cmd === 'one') this.zoomAt(this.oneToOne(), cx, cy, p);
      else if (cmd === 'two') this.zoomAt(this.oneToOne() * 2, cx, cy, p);
      else if (cmd === 'in') this.zoomAt(this.z * 1.5, cx, cy, p);
      else if (cmd === 'out') this.zoomAt(this.z / 1.5, cx, cy, p);
    }

    videos() { return this.panes.flatMap((p) => p.media).filter((m) => m.tagName === 'VIDEO'); }
    master() { const v = this.videos(); return v.length ? v[v.length - 1] : null; }
    paneAt(clientX) {
      if (this.panes.length < 2 || this.mode !== 'side') return this.panes[0];
      return this.panes.find((p) => { const r = p.el.getBoundingClientRect(); return clientX >= r.left && clientX <= r.right; }) || this.panes[0];
    }
    loading(on, msg) { this.loadingEl.classList.toggle('show', !!on); if (msg) this.loadingEl.querySelector('.msg').textContent = msg; }

    clear() {
      cancelAnimationFrame(this.raf);
      this.videos().forEach((v) => { try { v.pause(); v.removeAttribute('src'); v.load(); } catch (e) { /* ignore */ } });
      this.panes.forEach((p) => p.el.remove());
      this.st.querySelectorAll('.gh-divider,.gh-lab').forEach((n) => n.remove());
      this.panes = []; this.items = [];
      this.tp.classList.remove('show');
    }

    // items: [{url, kind, label, color}], mode: single | wipe | side
    async load(items, mode = 'single', opts = {}) {
      const token = ++this.token;
      this.clear();
      this.items = items.filter((i) => i && i.url);
      this.mode = this.items.length < 2 ? 'single' : mode;
      if (!this.items.length) { this.loading(true, opts.empty || 'No preview available'); return; }
      this.loading(true, opts.msg || 'Loading…');
      const keepView = opts.keepView && this.lastView;
      const mk = (it) => {
        const el = document.createElement(it.kind === 'video' ? 'video' : 'img');
        if (it.kind === 'video') { el.playsInline = true; el.preload = 'auto'; el.loop = this.loop; el.muted = true; el.crossOrigin = null; }
        else { el.decoding = 'async'; el.draggable = false; }
        el.src = U(it.url);
        return el;
      };
      const ready = (el) => new Promise((res) => {
        const ok = () => res(true);
        if (el.tagName === 'VIDEO') { if (el.readyState >= 1) ok(); else { el.addEventListener('loadedmetadata', ok, { once: true }); el.addEventListener('error', () => res(false), { once: true }); } }
        else if (el.complete && el.naturalWidth) ok(); else { el.addEventListener('load', ok, { once: true }); el.addEventListener('error', () => res(false), { once: true }); }
        setTimeout(() => res(false), 20000);
      });
      const nat = (el, it) => (el.tagName === 'VIDEO' ? [el.videoWidth || it.w || 16, el.videoHeight || it.h || 9] : [el.naturalWidth || it.w || 16, el.naturalHeight || it.h || 9]);

      const mkPane = (cls) => { const el = document.createElement('div'); el.className = 'gh-pane ' + cls; this.st.insertBefore(el, this.tag); return el; };
      const medias = this.items.map(mk);
      if (this.mode === 'side') {
        this.items.forEach((it, i) => {
          const el = mkPane('side' + i); const layer = document.createElement('div'); layer.className = 'gh-layer';
          const frame = document.createElement('div'); frame.className = 'gh-frame'; frame.appendChild(medias[i]); layer.appendChild(frame); el.appendChild(layer);
          this.panes.push({ el, frames: [frame], media: [medias[i]], items: [it] });
          el.insertAdjacentHTML('beforeend', `<div class="gh-lab l" style="max-width:calc(100% - 24px)"><span class="c" style="--c:${it.color || '#94a3b8'}"></span>${esc(it.label)}</div>`);
        });
      } else {
        const el = mkPane('one');
        const frames = this.items.map((it, i) => {
          const layer = document.createElement('div'); layer.className = 'gh-layer' + (i === 1 ? ' b' : '');
          const frame = document.createElement('div'); frame.className = 'gh-frame'; frame.appendChild(medias[i]);
          layer.appendChild(frame); el.appendChild(layer); return frame;
        });
        this.panes.push({ el, frames, media: medias, items: this.items });
        if (this.mode === 'wipe') {
          el.insertAdjacentHTML('beforeend', `<div class="gh-divider"><div class="gh-handle">⇆</div></div>
            <div class="gh-lab l"><span class="c" style="--c:${this.items[0].color || '#94a3b8'}"></span>${esc(this.items[0].label)}</div>
            <div class="gh-lab r"><span class="c" style="--c:${this.items[1].color || '#94a3b8'}"></span>${esc(this.items[1].label)}</div>`);
        }
      }
      const oks = await Promise.all(medias.map(ready));
      if (token !== this.token) return;
      this.panes.forEach((p) => { p.nat = p.media.map((m, i) => nat(m, p.items[i])); });
      this.layout();
      if (keepView) { this.z = this.lastView.z; this.fx = this.lastView.fx; this.fy = this.lastView.fy; this.clamp(); this.apply(); } else this.reset();
      this.loading(!oks.some(Boolean), 'This file could not be shown in the browser.');
      const vids = this.videos();
      this.tp.classList.toggle('show', vids.length > 0);
      if (vids.length) {
        const m = this.master();
        vids.forEach((v) => { v.muted = v !== m || this.muted; v.loop = this.loop; v.playbackRate = +this.tp.querySelector('[data-t="rate"]').value; });
        this.tp.querySelector('[data-t="mute"]').textContent = this.muted ? '🔇' : '🔊';
        m.addEventListener('play', () => this.syncLoop());
        m.addEventListener('timeupdate', () => this.updTime());
        m.addEventListener('seeked', () => this.updTime());
        m.addEventListener('pause', () => this.updPlay());
        m.addEventListener('ended', () => { if (this.loop) { vids.forEach((v) => { v.currentTime = 0; }); this.play(); } else this.updPlay(); });
        if (opts.autoplay !== false) this.play(); else this.updPlay();
        this.updTime();
      }
    }

    layout() {
      if (!this.panes.length) return;
      const W = this.st.clientWidth; const H = this.st.clientHeight;
      const gap = 6;
      this.panes.forEach((p, i) => {
        const pw = this.mode === 'side' ? (W - gap) / 2 : W;
        p.el.style.left = (this.mode === 'side' ? i * (pw + gap) : 0) + 'px';
        p.el.style.width = pw + 'px';
        p.pw = pw; p.ph = H;
        // fit rectangle: the output (last item) decides the frame; the others are drawn into the same frame
        const ref = p.nat ? p.nat[p.nat.length - 1] : [16, 9];
        const s = Math.min(pw / ref[0], H / ref[1]);
        const w = ref[0] * s; const h = ref[1] * s;
        p.fit = { l: (pw - w) / 2, t: (H - h) / 2, w, h, s };
        p.frames.forEach((fr, j) => {
          fr.style.left = p.fit.l + 'px'; fr.style.top = p.fit.t + 'px'; fr.style.width = w + 'px'; fr.style.height = h + 'px';
          const n = p.nat ? p.nat[j] : ref;
          fr.classList.toggle('contain', Math.abs(n[0] / n[1] - ref[0] / ref[1]) > 0.03);
        });
      });
      this.clamp(); this.apply();
    }

    oneToOne() {
      const p = this.panes[0];
      if (!p || !p.nat) return 1;
      const n = p.nat[p.nat.length - 1];
      return Math.max(1.0001, n[0] / p.fit.w);
    }
    reset() { this.z = 1; this.fx = 0; this.fy = 0; this.apply(); }
    zoomAt(z2, cx, cy, pane) {
      pane = pane || this.panes[0];
      if (!pane || !pane.fit) return;
      z2 = Math.min(64, Math.max(1, z2));
      const r = pane.el.getBoundingClientRect();
      const px = cx - r.left; const py = cy - r.top;
      const f = pane.fit;
      const u = (px - f.l - this.fx * f.w) / (f.w * this.z);
      const v = (py - f.t - this.fy * f.h) / (f.h * this.z);
      this.z = z2;
      this.fx = (px - f.l - u * f.w * z2) / f.w;
      this.fy = (py - f.t - v * f.h * z2) / f.h;
      this.clamp(); this.apply();
    }
    clamp() {
      const p = this.panes[0];
      if (!p || !p.fit) return;
      const f = p.fit;
      const fix = (pos, len, view, off) => {
        const size = len * this.z;
        if (size <= view + 0.5) return ((view - size) / 2 - off) / len;
        return Math.min(-off / len, Math.max((view - off - size) / len, pos));
      };
      if (this.z <= 1.0001) { this.fx = 0; this.fy = 0; return; }
      this.fx = fix(this.fx, f.w, p.pw, f.l);
      this.fy = fix(this.fy, f.h, p.ph, f.t);
    }
    apply() {
      this.panes.forEach((p) => {
        if (!p.fit) return;
        const tf = `translate(${this.fx * p.fit.w}px, ${this.fy * p.fit.h}px) scale(${this.z})`;
        p.frames.forEach((fr) => { fr.style.transform = tf; });
        if (this.mode === 'wipe') {
          const b = p.el.querySelector('.gh-layer.b');
          if (b) b.style.clipPath = `inset(0 0 0 ${this.split * 100}%)`;
          const d = p.el.querySelector('.gh-divider');
          if (d) d.style.left = (this.split * 100) + '%';
        }
      });
      this.lastView = { z: this.z, fx: this.fx, fy: this.fy };
      const p = this.panes[0];
      const eff = p && p.nat ? (this.z * p.fit.w) / p.nat[p.nat.length - 1][0] : 1;
      this.st.classList.toggle('pan', this.z > 1.001);
      this.st.classList.toggle('pixel', eff >= 2.5);
      this.tag.textContent = (this.z <= 1.001 ? 'Fit · ' : '') + Math.round(eff * 100) + '%';
      this.tag.classList.add('show');
      clearTimeout(this.tagT);
      this.tagT = setTimeout(() => this.tag.classList.remove('show'), 1100);
    }
    setSplitFromX(clientX) {
      const p = this.panes[0];
      const r = p.el.getBoundingClientRect();
      this.split = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      this.apply();
    }
    nudgeSplit(d) { if (this.mode === 'wipe') { this.split = Math.min(1, Math.max(0, this.split + d)); this.apply(); } }

    // ---- video transport
    play() { const vids = this.videos(); const m = this.master(); vids.forEach((v) => { if (v !== m && m) { try { v.currentTime = m.currentTime; } catch (e) { /* */ } } v.play().catch(() => {}); }); this.updPlay(); }
    pause() { this.videos().forEach((v) => v.pause()); this.updPlay(); }
    toggle() { const m = this.master(); if (!m) return; if (m.paused) this.play(); else this.pause(); }
    step(dir) {
      const m = this.master(); if (!m) return;
      this.pause();
      const fps = (this.items[this.items.length - 1] && this.items[this.items.length - 1].fps) || 30;
      const t = Math.min(m.duration || 0, Math.max(0, m.currentTime + dir / fps));
      this.videos().forEach((v) => { try { v.currentTime = Math.min(t, (v.duration || t)); } catch (e) { /* */ } });
      setTimeout(() => this.updTime(), 60);
    }
    toggleMute() { this.muted = !this.muted; LS.set('muted', this.muted); const m = this.master(); if (m) m.muted = this.muted; this.tp.querySelector('[data-t="mute"]').textContent = this.muted ? '🔇' : '🔊'; }
    updPlay() { const m = this.master(); this.tp.querySelector('[data-t="play"]').textContent = m && !m.paused ? '⏸' : '▶'; }
    updTime() {
      const m = this.master(); if (!m) return;
      const d = m.duration || 0;
      this.timeEl.textContent = `${fmtClock(m.currentTime)} / ${fmtClock(d)}`;
      if (document.activeElement !== this.scrub) this.scrub.value = d ? Math.round((m.currentTime / d) * 1000) : 0;
    }
    syncLoop() {
      cancelAnimationFrame(this.raf);
      this.updPlay();
      const tick = () => {
        const m = this.master();
        if (!m) return;
        this.updTime();
        if (!m.paused) {
          this.videos().forEach((v) => {
            if (v === m) return;
            if (v.paused && !m.paused) v.play().catch(() => {});
            if (Math.abs(v.currentTime - m.currentTime) > 0.09 && v.duration && m.currentTime < v.duration) { try { v.currentTime = m.currentTime; } catch (e) { /* */ } }
          });
          this.raf = requestAnimationFrame(tick);
        }
      };
      this.raf = requestAnimationFrame(tick);
    }
  }

  // ================================================================== VIEWER
  const V = { open: false, list: [], idx: 0, it: null, mode: LS.get('vmode', 'out'), details: null, view: null };
  const vEl = portal('gh-ov gh-viewer');
  vEl.innerHTML = `
    <div class="gh-vtop">
      <button class="gh-btn icon" data-v="prev" title="Previous (←)">‹</button>
      <button class="gh-btn icon" data-v="next" title="Next (→)">›</button>
      <div class="gh-vtitle"><span class="gh-badge"></span><div style="min-width:0"><div class="gh-vname"></div><div class="gh-vsub"></div></div></div>
      <div class="gh-vtools">
        <div class="gh-seg" data-vm><button data-m="out" title="The output (O)">Output</button><button data-m="wipe" title="Input and output with a slider (W)">⇆ Input vs output</button><button data-m="side" title="Input and output side by side (S)">▥ Side by side</button></div>
        <button class="gh-btn icon" data-v="fav" title="Favorite (F)">☆</button>
        <button class="gh-btn icon" data-v="dl" title="Download (D)">⬇</button>
        <button class="gh-btn icon" data-v="panel" title="Details panel (I)">ⓘ</button>
        <button class="gh-btn icon" data-v="fs" title="Full screen (F11 style, Esc leaves)">⛶</button>
        <button class="gh-btn icon" data-v="close" title="Close (Esc)">✕</button>
      </div>
    </div>
    <div class="gh-vbody"><div class="gh-stagewrap"></div><aside class="gh-panel"></aside></div>`;
  const vStage = new Stage(vEl.querySelector('.gh-stagewrap'));
  const vPanel = vEl.querySelector('.gh-panel');
  if (LS.get('panel', true) === false) vEl.classList.add('nopanel');

  function openViewer(id) {
    stopHover();
    V.list = S.filtered.map((i) => i.id);
    V.idx = Math.max(0, V.list.indexOf(id));
    if (!V.list.length) V.list = [id];
    V.open = true;
    vEl.classList.add('show');
    document.documentElement.style.overflow = 'hidden';
    showViewerItem();
  }
  function closeViewer() {
    if (!V.open) return;
    V.open = false;
    vStage.clear();
    vEl.classList.remove('show');
    document.documentElement.style.overflow = '';
    const card = content.querySelector(`.gh-card[data-id="${V.it && V.it.id}"]`);
    if (card) card.focus({ preventScroll: false });
  }
  function viewerFavState() { const b = vEl.querySelector('[data-v="fav"]'); b.textContent = V.it.fav ? '★' : '☆'; b.classList.toggle('on', !!V.it.fav); }
  async function playable(it, which) {
    const url = which === 'inp' ? it.in_url : it.url;
    const kind = which === 'inp' ? it.in_kind : it.kind;
    if (kind !== 'video' || (which === 'out' && it.playable)) return url;
    if (which === 'inp') { const r = await call('hist_playable', { id: it.id, which }); return (r && r.url) || url; }
    vStage.loading(true, 'Making a browser-friendly preview of this video… (first time only)');
    const r = await call('hist_playable', { id: it.id, which });
    return (r && r.url) || url;
  }
  async function showViewerItem(keepView = false) {
    const it = S.byId.get(V.list[V.idx]);
    if (!it) { closeViewer(); return; }
    V.it = it;
    const badge = vEl.querySelector('.gh-vtitle .gh-badge');
    badge.style.setProperty('--c', it.color);
    badge.className = 'gh-badge gh-eng';
    badge.textContent = `${it.icon} ${it.label}`;
    vEl.querySelector('.gh-vname').textContent = it.name || '(no output)';
    vEl.querySelector('.gh-vsub').textContent = [fmtRes(it.w, it.h) + (it.scale && it.scale !== 1 ? ` · ${it.scale}× upscale` : ''), it.kind === 'video' ? fmtDur(it.dur) : '', relTime(it.ts), `${V.idx + 1} / ${V.list.length}`].filter(Boolean).join('  ·  ');
    vEl.querySelector('[data-v="prev"]').disabled = V.idx <= 0;
    vEl.querySelector('[data-v="next"]').disabled = V.idx >= V.list.length - 1;
    viewerFavState();
    const canCmp = it.in_exists && it.in_url && it.out;
    vEl.querySelectorAll('[data-vm] button').forEach((b) => { b.disabled = b.dataset.m !== 'out' && !canCmp; b.title = b.disabled ? 'The input file of this run is no longer on disk' : b.title; });
    const mode = canCmp ? V.mode : 'out';
    vEl.querySelectorAll('[data-vm] button').forEach((b) => b.classList.toggle('on', b.dataset.m === mode));
    renderPanel(it, null);
    loadViewerStage(it, mode, keepView);
    const d = await call('hist_details', { id: it.id });
    if (V.it === it && d) { V.details = d; renderPanel(it, d); }
  }
  async function loadViewerStage(it, mode, keepView) {
    if (!it.out) { vStage.clear(); vStage.loading(true, `No output file · ${statusName[it.status] || ''}`); return; }
    const out = { url: await playable(it, 'out'), kind: it.kind, label: `${it.icon} ${it.label} · ${fmtRes(it.w, it.h)}`, color: it.color, fps: it.fps, w: it.w, h: it.h };
    if (V.it !== it) return;
    if (mode === 'out') { vStage.load([out], 'single', { keepView }); return; }
    const inp = { url: await playable(it, 'inp'), kind: it.in_kind, label: `Input · ${fmtRes(it.in_w, it.in_h) || 'original'}`, color: '#94a3b8', w: it.in_w, h: it.in_h };
    if (V.it !== it) return;
    vStage.load([inp, out], mode, { keepView });
  }
  async function showStep(it, steps, i) {
    const s = steps[i];
    const prev = i > 0 ? steps.slice(0, i).reverse().find((x) => x.url) : null;
    const before = prev ? { url: prev.url, kind: prev.kind, label: `Step ${prev.index} · ${prev.label}`, color: '#94a3b8' } : (it.in_exists ? { url: await playable(it, 'inp'), kind: it.in_kind, label: 'Chain input', color: '#94a3b8' } : null);
    const after = { url: s.url, kind: s.kind, label: `Step ${s.index} · ${s.icon} ${s.label}`, color: (S.engines[s.engine] || {}).color };
    vPanel.querySelectorAll('.gh-step').forEach((el) => el.classList.toggle('active', +el.dataset.i === i));
    vStage.load(before ? [before, after] : [after], before ? 'wipe' : 'single');
  }

  function factsHTML(it) {
    const facts = [
      ['Engine', `<span class="gh-pill" style="--c:${it.color}"><i></i>${esc(it.icon)} ${esc(it.label)}</span>`],
      ['Model', esc(it.model || '—')],
      ['Output', esc(fmtRes(it.w, it.h) || '—')],
      ['Input', esc(fmtRes(it.in_w, it.in_h) || (it.inp ? (it.in_exists ? '—' : 'not on disk') : '—'))],
    ];
    if (it.scale) facts.push(['Upscale', `${it.scale}×`]);
    if (it.kind === 'video') facts.push(['Length', esc(fmtDur(it.dur) || '—')], ['Frame rate', it.fps ? `${it.fps} fps` : '—'], ['Frames', it.frames ? String(it.frames) : '—'], ['Codec', esc((it.codec || '—') + (it.audio ? ' + audio' : ''))]);
    facts.push(['File size', esc(fmtSize(it.size) || '—')]);
    if (it.elapsed != null) facts.push(['Run time', esc(fmtElapsed(it.elapsed))]);
    facts.push(['Status', esc(statusName[it.status] || it.status)]);
    facts.push(['Created', esc(fullDate(it.ts)), 'wide']);
    if (it.status_text && it.status !== 'ok') facts.push(['Message', esc(it.status_text), 'wide wrapv']);
    return facts.map(([k, v, w]) => `<div class="gh-fact ${w ? 'wide' : ''}"><div class="k">${k}</div><div class="v ${w && w.includes('wrapv') ? 'wrap' : ''}">${v}</div></div>`).join('');
  }

  function renderPanel(it, d) {
    const isChain = !!it.chain_run;
    const canReload = isChain || it.has_args;
    const tabLabel = (Object.values(S.engines).find((e) => e.tab === it.tab) || {}).label || it.label;
    const reloadTitle = isChain ? 'Opens the Chains tab with these steps and this input' : `Opens ${tabLabel} with every setting of this run and its input`;
    let html = `<div class="gh-sec gh-actions">
        <button class="gh-btn primary" data-p="reload" ${canReload ? '' : 'disabled'} title="${esc(reloadTitle)} (R)"><span class="gh-ico">↻</span>${isChain ? 'Reload this chain' : 'Reload parameters'}</button>
        <div class="gh-reload-note">${canReload ? esc(reloadTitle) : 'No saved settings for this file'}</div>
        <div class="row2"><button class="gh-btn" data-p="send" ${it.out ? '' : 'disabled'}><span class="gh-ico">➜</span>Send output to…</button>
          <button class="gh-btn" data-p="pick" ${it.out ? '' : 'disabled'}><span class="gh-ico">⚖️</span>Compare with…</button></div>
        <div class="row2"><button class="gh-btn" data-p="dl" ${it.out ? '' : 'disabled'}><span class="gh-ico">⬇</span>Download</button>
          <button class="gh-btn" data-p="folder"><span class="gh-ico">📂</span>Show in folder</button></div>
        <div class="row2"><button class="gh-btn" data-p="cmpvid" ${it.cmp_url ? '' : 'disabled'} title="The side-by-side comparison video the run made"><span class="gh-ico">🎞️</span>Comparison video</button>
          <button class="gh-btn danger" data-p="del"><span class="gh-ico">🗑</span>Delete</button></div>
      </div>
      <div class="gh-sec"><h4>Details</h4><div class="gh-facts">${factsHTML(it)}</div></div>`;
    if (d && d.steps && d.steps.length) {
      html += `<div class="gh-sec"><h4>Chain steps · click to compare each step with the one before</h4><div class="gh-steps">` +
        d.steps.map((s, i) => `<div class="gh-step" data-i="${i}"><span class="num">${esc(s.index)}</span><div class="t">${esc(s.icon)} ${esc(s.label)}<small>${esc(s.status || '')}${s.w ? ' · ' + fmtRes(s.w, s.h) : ''}${s.seconds ? ' · ' + fmtElapsed(s.seconds) : ''}</small></div>${s.url ? '<button class="gh-btn" data-p="step">View</button>' : ''}</div>`).join('') + '</div></div>';
    }
    if (d && d.extras && d.extras.length) {
      html += `<div class="gh-sec"><h4>Other files of this run</h4><div class="gh-steps">` + d.extras.map((x, i) => `<div class="gh-step" data-x="${i}"><span class="num" style="background:#334155">${x.kind === 'video' ? '🎬' : '🖼️'}</span><div class="t">${esc(x.name)}</div><button class="gh-btn" data-p="extra">View</button></div>`).join('') + '</div></div>';
    }
    html += `<div class="gh-sec"><h4>Files</h4>
      ${it.out ? `<div class="gh-path"><span class="lab">OUT</span><code>${esc(it.out)}</code><button class="gh-btn" data-p="copyout">Copy</button></div>` : ''}
      ${it.inp ? `<div class="gh-path"><span class="lab">IN</span><code>${esc(it.inp)}${it.in_exists ? '' : ' (not on disk)'}</code><button class="gh-btn" data-p="copyin">Copy</button></div>` : ''}
      ${it.dir ? `<div class="gh-path"><span class="lab">RUN</span><code>${esc(it.dir)}</code><button class="gh-btn" data-p="copydir">Copy</button></div>` : ''}</div>`;
    if (d && d.groups && d.groups.length) {
      const n = d.groups.reduce((a, g) => a + g.items.length, 0);
      html += `<div class="gh-sec"><h4>Parameters · ${n}</h4><input class="gh-psearch" placeholder="Filter parameters…"><div class="gh-pgroups">` +
        d.groups.map((g, gi) => `<details class="gh-pgroup" ${gi < 3 ? 'open' : ''}><summary>${esc(g.name)} <span>${g.items.length}</span></summary><div class="gh-kv">${g.items.map(([k, v]) => `<div class="k" data-k="${esc(k.toLowerCase())}">${esc(k.replace(/_/g, ' '))}</div><div class="v ${String(v).length > 60 ? 'long' : ''}" data-k="${esc(k.toLowerCase())}">${esc(v)}</div>`).join('')}</div></details>`).join('') +
        `</div><button class="gh-btn" data-p="copyjson" style="margin-top:6px"><span class="gh-ico">{ }</span>Copy all as JSON</button></div>`;
    } else if (d && !isChain) {
      html += `<div class="gh-sec"><h4>Parameters</h4><div style="color:#8190ab;font-size:12.5px">This file was made without a saved settings record (older run, an external batch folder or a cancelled job).</div></div>`;
    } else if (!d) {
      html += '<div class="gh-sec"><h4>Parameters</h4><div style="color:#8190ab;font-size:12.5px">Loading…</div></div>';
    }
    vPanel.innerHTML = html;
    const ps = vPanel.querySelector('.gh-psearch');
    if (ps) ps.addEventListener('input', () => {
      const q = ps.value.trim().toLowerCase();
      vPanel.querySelectorAll('.gh-kv [data-k]').forEach((el) => { el.style.display = !q || el.dataset.k.includes(q) || el.textContent.toLowerCase().includes(q) ? '' : 'none'; });
      vPanel.querySelectorAll('.gh-pgroup').forEach((g) => { if (q) g.open = true; });
    });
  }

  vPanel.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-p]');
    const it = V.it;
    if (!b || !it) return;
    const p = b.dataset.p;
    if (p === 'reload') sendAction('reload', it);
    else if (p === 'send') sendMenu(b, it);
    else if (p === 'pick') startPick(it.id);
    else if (p === 'dl') download(it.url, it.name);
    else if (p === 'folder') { const r = await call('hist_open', { id: it.id, what: 'select' }); if (r) toast(r.msg, r.ok ? 'ok' : 'bad'); }
    else if (p === 'cmpvid') { vEl.querySelectorAll('[data-vm] button').forEach((x) => x.classList.remove('on')); vStage.load([{ url: it.cmp_url, kind: 'video', label: 'Comparison video', color: it.color }], 'single'); }
    else if (p === 'del') { const idx = V.idx; const ok = await deleteItems([it.id]); if (ok) { V.list.splice(idx, 1); if (!V.list.length) closeViewer(); else { V.idx = Math.min(idx, V.list.length - 1); showViewerItem(); } } }
    else if (p === 'copyout') copyText(it.out);
    else if (p === 'copyin') copyText(it.inp);
    else if (p === 'copydir') copyText(it.dir);
    else if (p === 'copyjson' && V.details) copyText(JSON.stringify(V.details.args, null, 2));
    else if (p === 'step' && V.details) showStep(it, V.details.steps, +b.closest('.gh-step').dataset.i);
    else if (p === 'extra' && V.details) { const x = V.details.extras[+b.closest('.gh-step').dataset.x]; vStage.load([{ url: x.url, kind: x.kind, label: x.name, color: it.color }], 'single'); }
  });
  vEl.querySelector('.gh-vtop').addEventListener('click', (e) => {
    const m = e.target.closest('[data-m]');
    if (m && !m.disabled) { setViewerMode(m.dataset.m); return; }
    const b = e.target.closest('[data-v]');
    if (!b) return;
    const a = b.dataset.v;
    if (a === 'close') closeViewer();
    else if (a === 'prev') navViewer(-1);
    else if (a === 'next') navViewer(1);
    else if (a === 'fav') toggleFav([V.it.id]);
    else if (a === 'dl' && V.it.url) download(V.it.url, V.it.name);
    else if (a === 'fs') toggleFullscreen(vEl);
    else if (a === 'panel') { vEl.classList.toggle('nopanel'); LS.set('panel', !vEl.classList.contains('nopanel')); setTimeout(() => vStage.layout(), 30); }
  });
  function setViewerMode(m) {
    const it = V.it;
    if (!it || (m !== 'out' && !(it.in_exists && it.in_url))) return;
    V.mode = m; LS.set('vmode', m);
    vEl.querySelectorAll('[data-vm] button').forEach((b) => b.classList.toggle('on', b.dataset.m === m));
    loadViewerStage(it, m, true);
  }
  function navViewer(d) {
    const n = V.idx + d;
    if (n < 0 || n >= V.list.length) return;
    V.idx = n;
    showViewerItem(true);
  }

  // ================================================================== COMPARE
  const C = { open: false, a: null, b: null, mode: 'wipe', only: true, da: null, db: null };
  const cEl = portal('gh-ov gh-compare');
  cEl.innerHTML = `
    <div class="gh-vtop">
      <div class="gh-vtitle" style="gap:14px"><div class="gh-cmpside" data-side="a"></div><button class="gh-btn icon" data-c="swap" title="Swap sides (X)">⇄</button><div class="gh-cmpside" data-side="b"></div></div>
      <div class="gh-vtools">
        <div class="gh-seg" data-cm><button data-m="wipe" title="Slider (W)">⇆ Slider</button><button data-m="side" title="Side by side (S)">▥ Side by side</button></div>
        <button class="gh-btn" data-c="diff" title="Show what is different (I)"><span class="gh-ico">≠</span>Differences</button>
        <button class="gh-btn icon" data-c="fs" title="Full screen">⛶</button>
        <button class="gh-btn icon" data-c="close" title="Close (Esc)">✕</button>
      </div>
    </div>
    <div class="gh-cmp-body"><div class="gh-stagewrap"></div><aside class="gh-diff"></aside></div>`;
  const cStage = new Stage(cEl.querySelector('.gh-stagewrap'));
  const cDiff = cEl.querySelector('.gh-diff');
  if (LS.get('diff', true) === false) cEl.classList.add('nodiff');

  async function openCompare(idA, idB) {
    const a = S.byId.get(idA); const b = S.byId.get(idB);
    if (!a || !b || !a.out || !b.out) { toast('Both generations need an output file to compare.', 'bad'); return; }
    closeViewer();
    C.a = a; C.b = b; C.open = true; C.da = null; C.db = null;
    cEl.classList.add('show');
    document.documentElement.style.overflow = 'hidden';
    loadCompare();
    const [da, db] = await Promise.all([call('hist_details', { id: a.id }), call('hist_details', { id: b.id })]);
    if (C.a === a && C.b === b) { C.da = da; C.db = db; renderDiff(); }
  }
  function closeCompare() {
    if (!C.open) return;
    C.open = false; cStage.clear(); cEl.classList.remove('show');
    document.documentElement.style.overflow = '';
    if (C.fromSel) { C.fromSel = false; clearSel(); }
  }
  async function loadCompare(keepView = false) {
    const side = (it) => `<span class="gh-badge gh-eng" style="--c:${it.color}">${esc(it.icon)} ${esc(it.label)}</span><span class="nm" title="${esc(it.out)}">${esc(it.name)}</span>`;
    cEl.querySelector('[data-side="a"]').innerHTML = side(C.a);
    cEl.querySelector('[data-side="b"]').innerHTML = side(C.b);
    const mixed = C.a.kind !== C.b.kind;
    const mode = mixed ? 'side' : C.mode;
    cEl.querySelectorAll('[data-cm] button').forEach((x) => { x.classList.toggle('on', x.dataset.m === mode); x.disabled = mixed && x.dataset.m === 'wipe'; });
    const mk = async (it, tag) => ({ url: it.kind === 'video' && !it.playable ? ((await call('hist_playable', { id: it.id, which: 'out' })) || {}).url || it.url : it.url, kind: it.kind, label: `${tag} · ${it.label} · ${fmtRes(it.w, it.h)}`, color: it.color, fps: it.fps, w: it.w, h: it.h });
    if ((C.a.kind === 'video' && !C.a.playable) || (C.b.kind === 'video' && !C.b.playable)) cStage.loading(true, 'Making a browser-friendly preview… (first time only)');
    const A = await mk(C.a, 'A'); const B = await mk(C.b, 'B');
    if (!C.open) return;
    cStage.load([A, B], mode, { keepView });
    renderDiff();
  }
  function renderDiff() {
    const a = C.a; const b = C.b;
    const facts = [
      ['Engine', a.label, b.label], ['Model', a.model, b.model], ['Resolution', fmtRes(a.w, a.h), fmtRes(b.w, b.h)],
      ['Upscale', a.scale ? a.scale + '×' : '', b.scale ? b.scale + '×' : ''], ['Length', fmtDur(a.dur), fmtDur(b.dur)],
      ['Frame rate', a.fps || '', b.fps || ''], ['Frames', a.frames || '', b.frames || ''], ['File size', fmtSize(a.size), fmtSize(b.size)],
      ['Run time', fmtElapsed(a.elapsed), fmtElapsed(b.elapsed)], ['Created', fullDate(a.ts), fullDate(b.ts)],
      ['Input file', (a.inp || '').split(/[\\/]/).pop(), (b.inp || '').split(/[\\/]/).pop()],
    ];
    const pa = (C.da && C.da.args) || {}; const pb = (C.db && C.db.args) || {};
    const keys = [...new Set([...Object.keys(pa), ...Object.keys(pb)])].sort();
    const fmt = (v) => (v === undefined ? '—' : typeof v === 'boolean' ? (v ? 'On' : 'Off') : String(v));
    const params = keys.map((k) => [k.replace(/_/g, ' '), fmt(pa[k]), fmt(pb[k])]);
    const nDiff = params.filter((r) => r[1] !== r[2]).length;
    const row = ([k, x, y]) => { const d = String(x) !== String(y); if (!d && C.only) return ''; return `<tr class="${d ? 'd' : 'same'}"><td class="k">${esc(k)}</td><td class="a">${esc(x || '—')}</td><td class="b">${esc(y || '—')}</td></tr>`; };
    const loading = !C.da || !C.db;
    cDiff.innerHTML = `<div class="gh-sec"><h4>What is different</h4>
      <label style="display:flex;gap:8px;align-items:center;font-size:12.5px;color:#a8b3c7;margin-bottom:10px;cursor:pointer"><input type="checkbox" class="gh-only" ${C.only ? 'checked' : ''}> Only show what differs</label>
      <table class="gh-difft"><thead><tr><th>Item</th><th>A</th><th>B</th></tr></thead><tbody>${facts.map(row).join('')}</tbody></table></div>
      <div class="gh-sec" style="margin-top:16px"><h4>Parameters ${loading ? '· loading…' : `· ${nDiff} of ${keys.length} differ`}</h4>
      ${keys.length ? `<table class="gh-difft"><thead><tr><th>Setting</th><th>A</th><th>B</th></tr></thead><tbody>${params.map(row).join('') || '<tr><td colspan="3" style="color:#8190ab">Every saved setting is the same.</td></tr>'}</tbody></table>` : (loading ? '' : '<div style="color:#8190ab;font-size:12.5px">No saved settings for these files.</div>')}</div>`;
    cDiff.querySelector('.gh-only').addEventListener('change', (ev) => { C.only = ev.target.checked; renderDiff(); });
  }
  cEl.querySelector('.gh-vtop').addEventListener('click', (e) => {
    const m = e.target.closest('[data-m]');
    if (m && !m.disabled) { C.mode = m.dataset.m; loadCompare(true); return; }
    const b = e.target.closest('[data-c]');
    if (!b) return;
    if (b.dataset.c === 'close') closeCompare();
    else if (b.dataset.c === 'swap') { [C.a, C.b] = [C.b, C.a]; [C.da, C.db] = [C.db, C.da]; loadCompare(true); }
    else if (b.dataset.c === 'fs') toggleFullscreen(cEl);
    else if (b.dataset.c === 'diff') { cEl.classList.toggle('nodiff'); LS.set('diff', !cEl.classList.contains('nodiff')); setTimeout(() => cStage.layout(), 30); }
  });

  // ================================================================== keyboard
  document.addEventListener('keydown', (e) => {
    if (modal.classList.contains('show')) { if (e.key === 'Escape' && modalDone) modalDone(false); if (e.key === 'Enter' && modalDone) modalDone(true); return; }
    if (menu.classList.contains('show') && e.key === 'Escape') { closeMenu(); return; }
    const typing = isTyping(e.target);
    const stageKeys = (stg) => {
      if (e.key === ' ') { e.preventDefault(); stg.toggle(); return true; }
      if (e.key === ',') { stg.step(-1); return true; }
      if (e.key === '.') { stg.step(1); return true; }
      if (e.key === '0') { stg.zoomCmd('fit'); return true; }
      if (e.key === '1') { stg.zoomCmd('one'); return true; }
      if (e.key === '2') { stg.zoomCmd('two'); return true; }
      if (e.key === '+' || e.key === '=') { stg.zoomCmd('in'); return true; }
      if (e.key === '-') { stg.zoomCmd('out'); return true; }
      if (e.key === 'm' || e.key === 'M') { stg.toggleMute(); return true; }
      if (e.key === '[') { stg.nudgeSplit(-0.05); return true; }
      if (e.key === ']') { stg.nudgeSplit(0.05); return true; }
      return false;
    };
    if (C.open) {
      if (typing && e.key !== 'Escape') return;
      if (e.key === 'Escape') { closeCompare(); return; }
      if (stageKeys(cStage)) return;
      const k = e.key.toLowerCase();
      if (k === 'x') { [C.a, C.b] = [C.b, C.a]; [C.da, C.db] = [C.db, C.da]; loadCompare(true); }
      else if (k === 'w' && C.a.kind === C.b.kind) { C.mode = 'wipe'; loadCompare(true); }
      else if (k === 's') { C.mode = 'side'; loadCompare(true); }
      else if (k === 'i') cEl.querySelector('[data-c="diff"]').click();
      return;
    }
    if (V.open) {
      if (typing && e.key !== 'Escape') return;
      if (e.key === 'Escape') { closeViewer(); return; }
      if (e.key === 'ArrowLeft') { navViewer(-1); e.preventDefault(); return; }
      if (e.key === 'ArrowRight') { navViewer(1); e.preventDefault(); return; }
      if (stageKeys(vStage)) return;
      const k = e.key.toLowerCase();
      if (k === 'f') toggleFav([V.it.id]);
      else if (k === 'd' && V.it.url) download(V.it.url, V.it.name);
      else if (k === 'i') vEl.querySelector('[data-v="panel"]').click();
      else if (k === 'o') setViewerMode('out');
      else if (k === 'w') setViewerMode('wipe');
      else if (k === 's') setViewerMode('side');
      else if (k === 'r') { const b = vPanel.querySelector('[data-p="reload"]'); if (b && !b.disabled) b.click(); }
      else if (k === 'c') startPick(V.it.id);
      return;
    }
    if (!root.offsetParent) return;
    if (e.key === 'Escape') { if (S.pickFor) endPick(); else if (S.sel.size) clearSel(); else if (typing && e.target === searchInput) { searchInput.blur(); } return; }
    if (typing) return;
    if (e.key === '/') { e.preventDefault(); searchInput.focus(); searchInput.select(); }
    else if (e.key === '?') showShortcuts();
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a' && S.loaded && (root.contains(document.activeElement) || document.activeElement === document.body)) { e.preventDefault(); S.filtered.forEach((i) => S.sel.add(i.id)); render(true); }
  });

  // ================================================================== live updates
  let pollBusy = false;
  async function poll() {
    if (pollBusy || S.loading || !S.loaded || !root.offsetParent || document.visibilityState !== 'visible') return;
    pollBusy = true;
    try {
      const r = await server.hist_poll({});
      if (r && r.sig && r.sig !== S.sig) await refresh({ silent: true });
    } catch (e) { /* app restarting */ }
    pollBusy = false;
  }
  setInterval(poll, 4000);
  window.__ghOnShow = () => { if (!S.loaded) refresh(); else refresh({ silent: true }); };
  // first load when the tab is visible (Gradio mounts hidden tabs too)
  const firstIO = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting) && !S.loaded && !S.loading) refresh(); }, {});
  firstIO.observe(root);
  window.__ghDebug = { S, V, C, refresh };
})();
