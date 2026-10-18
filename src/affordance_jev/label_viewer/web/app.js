(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const dataset = window.UMDDataset;
  if (!dataset) {
    $('error').hidden = false;
    $('error').textContent = '无法读取数据索引，请保留 index.html 所在的完整目录后重新打开。';
    return;
  }
  const samples = dataset.samples;
  const classes = dataset.classes;
  const byId = new Map(classes.map(c => [c.id, c]));
  const fmt = n => n.toLocaleString('zh-CN');
  const PAGE_SIZE = 24;
  const canvases = ['rgbCanvas', 'originalCanvas', 'remappedCanvas', 'changedCanvas'].map($);
  const state = {filtered: [], index: 0, page: 0, token: 0, ready: false, row: null,
    rgb: null, oldLabels: null, newLabels: null, mode: 'overlay', opacity: .58,
    selectedClass: null, zoom: 1, center: [320, 240], fit: 'object', layers: []};
  const pending = new Map();
  const labelCache = new Map();
  let dragging = null;

  window.UMDLabelData = (id, runs) => {
    labelCache.set(id, runs);
    while (labelCache.size > 24) labelCache.delete(labelCache.keys().next().value);
    if (pending.has(id)) pending.get(id).resolve(runs);
  };

  function loadRuns(row) {
    if (labelCache.has(row.id)) return Promise.resolve(labelCache.get(row.id));
    if (pending.has(row.id)) return pending.get(row.id).promise;
    let resolve, reject;
    const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
    pending.set(row.id, {promise, resolve, reject});
    const script = document.createElement('script');
    script.src = `${row.path}/labels.js`;
    const timer = setTimeout(() => reject(new Error('读取标签超时')), 20000);
    script.onload = () => {
      if (!labelCache.has(row.id)) reject(new Error('标签文件未返回有效数据'));
    };
    script.onerror = () => reject(new Error('无法读取 labels.js，请确认 samples 文件夹完整'));
    document.head.append(script);
    return promise.finally(() => { clearTimeout(timer); script.remove(); pending.delete(row.id); });
  }

  function loadImage(path) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('无法读取 RGB，请确认 samples 文件夹完整'));
      image.src = path;
    });
  }

  function decode(runs, length) {
    const labels = new Int16Array(length);
    let offset = 0;
    if (runs.length % 2) throw new Error('标签编码长度错误');
    for (let i = 0; i < runs.length; i += 2) {
      const value = runs[i], count = runs[i + 1];
      if (!Number.isInteger(count) || count <= 0 || offset + count > length || value < -1 || value > 7)
        throw new Error('标签编码内容错误');
      labels.fill(value, offset, offset + count);
      offset += count;
    }
    if (offset !== length) throw new Error('标签尺寸不完整');
    return labels;
  }

  function setOptions(select, values, caption) {
    const previous = select.value;
    select.replaceChildren(new Option(caption, 'all'));
    for (const [value, title] of values) select.add(new Option(title, value));
    select.value = values.some(([v]) => v === previous) ? previous : 'all';
  }

  function updateOptions() {
    const split = $('split').value;
    const scoped = samples.filter(r => split === 'all' || r.split === split);
    const categories = [...new Set(scoped.map(r => r.category))].sort();
    setOptions($('category'), categories.map(c => [c, `${c} · ${dataset.categoryNames[c] || c}`]), '全部物体');
    const category = $('category').value;
    const instances = [...new Set(scoped.filter(r => category === 'all' || r.category === category).map(r => r.instance))].sort();
    setOptions($('instance'), instances.map(i => [i, i]), '全部实例');
  }

  function filterSamples(keepId = null) {
    const split = $('split').value, category = $('category').value, instance = $('instance').value;
    const query = $('search').value.trim().toLowerCase();
    state.filtered = samples.filter(r => (split === 'all' || r.split === split)
      && (category === 'all' || r.category === category)
      && (instance === 'all' || r.instance === instance)
      && (!$('changedOnly').checked || r.changed > 0)
      && (!query || `${r.id} ${r.category} ${dataset.categoryNames[r.category]} ${r.frame}`.toLowerCase().includes(query)));
    state.index = Math.max(0, state.filtered.findIndex(r => r.id === keepId));
    state.page = Math.floor(state.index / PAGE_SIZE);
    $('filterCount').textContent = `${fmt(state.filtered.length)} / ${fmt(samples.length)} 张`;
    renderList();
    if (state.filtered.length) showSample(state.index);
    else showEmpty();
  }

  function renderList() {
    const list = $('sampleList');
    list.replaceChildren();
    const start = state.page * PAGE_SIZE;
    for (let i = start; i < Math.min(start + PAGE_SIZE, state.filtered.length); i++) {
      const row = state.filtered[i];
      const card = document.createElement('button');
      card.className = 'sample-card' + (i === state.index ? ' active' : '');
      card.dataset.sampleId = row.id;
      card.setAttribute('aria-label', `${row.id}${row.changed ? '，有修改' : '，无修改'}`);
      card.setAttribute('aria-current', i === state.index ? 'true' : 'false');
      const image = document.createElement('img');
      image.src = `${row.path}/thumb.jpg`; image.loading = 'lazy'; image.alt = '';
      const text = document.createElement('span'); text.className = 'card-body';
      const title = document.createElement('strong'); title.textContent = row.instance;
      const meta = document.createElement('small');
      meta.textContent = `${String(row.frame).padStart(8, '0')} · ${row.split === 'test' ? '测试' : '训练'}`;
      text.append(title, meta); card.append(image, text);
      if (row.changed) {
        const dot = document.createElement('span'); dot.className = 'changed-dot';
        card.append(dot);
      }
      card.addEventListener('click', () => showSample(i));
      list.append(card);
    }
    if (!state.filtered.length) {
      const empty = document.createElement('p'); empty.className = 'empty-list';
      empty.textContent = '没有符合筛选条件的图像。可取消“只看有修改”，或点击“查看全部”。';
      list.append(empty);
    }
    const pages = Math.ceil(state.filtered.length / PAGE_SIZE);
    $('pageText').textContent = pages ? `${state.page + 1} / ${pages} 页` : '0 / 0 页';
    $('pagePrev').disabled = state.page === 0;
    $('pageNext').disabled = !pages || state.page >= pages - 1;
  }

  function clearCanvases() {
    for (const canvas of canvases) {
      const ctx = canvas.getContext('2d'); ctx.fillStyle = '#f5f7fa';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    document.querySelectorAll('.crosshair').forEach(c => { c.hidden = true; });
  }

  function showEmpty() {
    state.token++; state.ready = false; state.row = null;
    $('canvasGrid').setAttribute('aria-busy', 'false');
    $('error').hidden = true;
    $('sampleTitle').textContent = '没有符合条件的图像';
    $('sampleCategory').textContent = '调整左侧筛选条件';
    $('sampleMeta').textContent = '请调整筛选条件，或点击“查看全部”。';
    $('samplePosition').textContent = '0 / 0';
    $('previous').disabled = true; $('next').disabled = true;
    $('pixelTable').replaceChildren(); $('differenceNote').textContent = '';
    $('currentExplanation').textContent = '';
    $('footerStatus').textContent = '';
    $('inspector').textContent = '请选择一张图像。';
    for (const id of ['downloadRgb', 'downloadOriginal', 'downloadRemapped', 'downloadChanged']) $(id).removeAttribute('href');
    clearCanvases();
  }

  async function showSample(index) {
    if (index < 0 || index >= state.filtered.length) return;
    state.index = index; state.row = state.filtered[index]; state.ready = false;
    const row = state.row, token = ++state.token;
    state.page = Math.floor(index / PAGE_SIZE);
    renderList();
    const active = $('sampleList').querySelector('.active');
    if (active) $('sampleList').scrollTop = Math.max(0, active.offsetTop - $('sampleList').offsetTop - 90);
    $('sampleTitle').textContent = row.id;
    $('sampleCategory').textContent = `${dataset.categoryNames[row.category] || row.category} / ${row.category}`;
    $('sampleMeta').textContent = `novel/${row.split} · ${row.annotationType === 'manual' ? '人工标注' : '自动标注'} · 原始 ${row.size[0]} × ${row.size[1]} · ${row.changed ? '修改 ' + fmt(row.changed) + ' 像素' : '无修改'}`;
    $('samplePosition').textContent = `${index + 1} / ${fmt(state.filtered.length)}`;
    $('previous').disabled = index === 0; $('next').disabled = index === state.filtered.length - 1;
    $('canvasGrid').setAttribute('aria-busy', 'true'); $('error').hidden = true;
    $('footerStatus').textContent = '正在读取图像与标签…';
    $('inspector').textContent = '指向图像查看原始像素坐标与前后标签；放大后可拖动图像。';
    clearCanvases();
    $('downloadRgb').href = `${row.path}/rgb.jpg`;
    $('downloadOriginal').href = `${row.path}/original.png`;
    $('downloadRemapped').href = `${row.path}/ooal_rule.png`;
    $('downloadChanged').href = `${row.path}/changed.png`;
    for (const [id, name] of [['downloadRgb', 'rgb.jpg'], ['downloadOriginal', 'original.png'],
                             ['downloadRemapped', 'ooal_rule.png'], ['downloadChanged', 'changed.png']])
      $(id).download = row.id.replace('/', '_') + '_' + name;
    $('differenceNote').textContent = row.changed ? `洋红色：${fmt(row.changed)} 像素 · contain → scoop` : '无修改：两份 GT 逐像素一致';
    $('currentExplanation').textContent = row.changed
      ? `本图 ${fmt(row.changed)} 个像素仅改变功能类别，占物体标注区域的 ${(100 * row.changed / row.foreground).toFixed(1)}%。其余像素完全一致。`
      : '本图不触发 ladle 的 contain → scoop 规则，原始 GT 与修改后 GT 完全一致。';
    renderPixelTable(row);
    const hash = '#sample=' + encodeURIComponent(row.id);
    try { history.replaceState(null, '', hash); } catch (_) { /* file browsing may restrict history */ }
    try {
      const [rgb, runs] = await Promise.all([loadImage(`${row.path}/rgb.jpg`), loadRuns(row)]);
      if (token !== state.token) return;
      if (rgb.naturalWidth !== row.size[0] || rgb.naturalHeight !== row.size[1]) throw new Error('RGB 尺寸与标签不一致');
      const oldLabels = decode(runs, row.size[0] * row.size[1]);
      const newLabels = new Int16Array(oldLabels);
      let changed = 0;
      if (row.category === dataset.rule.object) {
        for (let i = 0; i < oldLabels.length; i++) if (oldLabels[i] === dataset.rule.from) {
          newLabels[i] = dataset.rule.to; changed++;
        }
      }
      if (changed !== row.changed) throw new Error('标签变化计数与索引不一致');
      state.rgb = rgb; state.oldLabels = oldLabels; state.newLabels = newLabels;
      state.ready = true;
      makeLayers(); setView(state.fit === 'image' ? 'image' : 'object');
      $('canvasGrid').setAttribute('aria-busy', 'false');
      $('footerStatus').textContent = '标签已校验 · 原始数据未改动';
    } catch (error) {
      if (token !== state.token) return;
      state.ready = false; $('canvasGrid').setAttribute('aria-busy', 'false');
      $('error').hidden = false; $('error').textContent = `${row.id}：${error.message}`;
      $('footerStatus').textContent = '读取失败';
    }
  }

  function renderPixelTable(row) {
    $('pixelTable').replaceChildren();
    for (const cls of classes) {
      const before = row.countsBefore[cls.id], after = row.countsAfter[cls.id];
      if (cls.id === 0 || before + after === 0) continue;
      const tr = document.createElement('tr'); tr.dataset.classId = cls.id;
      const title = document.createElement('td');
      const label = document.createElement('span'); label.className = 'table-class';
      const dot = document.createElement('i'); dot.style.setProperty('--color', `rgb(${cls.color.join(',')})`);
      label.append(dot, `${cls.id} · ${cls.name}`); title.append(label); tr.append(title);
      for (const value of [before, after]) { const td = document.createElement('td'); td.textContent = fmt(value); tr.append(td); }
      const delta = document.createElement('td'), diff = after - before;
      delta.textContent = diff > 0 ? '+' + fmt(diff) : diff < 0 ? '−' + fmt(-diff) : '—';
      delta.className = diff > 0 ? 'delta-positive' : diff < 0 ? 'delta-negative' : '';
      tr.append(delta); $('pixelTable').append(tr);
    }
  }

  function makeLayers() {
    if (!state.ready) return;
    const [width, height] = state.row.size;
    state.layers = [0, 1, 2].map(kind => {
      const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext('2d'), pixels = ctx.createImageData(width, height);
      const labels = kind === 0 ? state.oldLabels : state.newLabels;
      for (let i = 0; i < labels.length; i++) {
        const id = labels[i], changed = state.oldLabels[i] !== state.newLabels[i];
        if (kind === 2 ? !changed : id <= 0 || state.selectedClass !== null && id !== state.selectedClass) continue;
        const color = kind === 2 ? [218, 42, 137] : byId.get(id).color;
        pixels.data[i * 4] = color[0]; pixels.data[i * 4 + 1] = color[1];
        pixels.data[i * 4 + 2] = color[2]; pixels.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(pixels, 0, 0);
      return canvas;
    });
  }

  function viewRect() {
    const [width, height] = state.row.size, w = width / state.zoom, h = height / state.zoom;
    state.center[0] = Math.max(w / 2, Math.min(width - w / 2, state.center[0]));
    state.center[1] = Math.max(h / 2, Math.min(height - h / 2, state.center[1]));
    return {x: state.center[0] - w / 2, y: state.center[1] - h / 2, w, h};
  }

  function setView(mode) {
    state.fit = mode;
    $('fitObject').classList.toggle('active', mode === 'object');
    $('fitImage').classList.toggle('active', mode === 'image');
    if (!state.ready) return;
    const [width, height] = state.row.size, [x0, y0, x1, y1] = state.row.box;
    state.zoom = mode === 'image' ? 1 : Math.max(1, Math.min(12, width / Math.max(64, (x1-x0)*1.55), height / Math.max(64, (y1-y0)*1.55)));
    state.center = mode === 'image' ? [width / 2, height / 2] : [(x0+x1)/2, (y0+y1)/2];
    updateZoom(); drawPanels();
  }

  function updateZoom() {
    $('zoom').value = state.zoom;
    $('zoomValue').textContent = state.zoom.toFixed(1) + '×';
  }

  function drawPanels() {
    if (!state.ready) return;
    const rect = viewRect();
    canvases.forEach((canvas, i) => {
      const ctx = canvas.getContext('2d'); ctx.globalAlpha = 1;
      ctx.fillStyle = '#e8ebf0'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (i === 0 || i === 3 || state.mode === 'overlay') {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(state.rgb, rect.x, rect.y, rect.w, rect.h, 0, 0, canvas.width, canvas.height);
      }
      if (i > 0) {
        ctx.globalAlpha = i === 3 ? .78 : state.mode === 'labels' ? 1 : state.opacity;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(state.layers[i - 1], rect.x, rect.y, rect.w, rect.h, 0, 0, canvas.width, canvas.height);
      }
      ctx.globalAlpha = 1;
    });
  }

  function renderLegend() {
    $('legend').replaceChildren();
    const hint = document.createElement('span'); hint.className = 'legend-hint'; hint.textContent = '功能类别';
    $('legend').append(hint);
    for (const cls of [{id: null, name: '全部'}, ...classes.filter(c => c.id > 0)]) {
      const button = document.createElement('button');
      button.className = state.selectedClass === cls.id ? 'active' : '';
      button.setAttribute('aria-pressed', state.selectedClass === cls.id ? 'true' : 'false');
      button.dataset.classId = cls.id === null ? 'all' : cls.id;
      if (cls.color) {
        const dot = document.createElement('i'); dot.style.setProperty('--color', `rgb(${cls.color.join(',')})`);
        button.append(dot); button.title = `${cls.name_zh} · ID ${cls.id}；仅筛选显示，不修改标签`;
      }
      button.append(cls.name);
      button.addEventListener('click', () => {
        state.selectedClass = cls.id === state.selectedClass ? null : cls.id;
        renderLegend(); makeLayers(); drawPanels();
      });
      $('legend').append(button);
    }
  }

  function inspect(event, canvas) {
    if (!state.ready) return;
    const bounds = canvas.getBoundingClientRect(), rect = viewRect();
    const u = Math.max(0, Math.min(.999999, (event.clientX - bounds.left) / bounds.width));
    const v = Math.max(0, Math.min(.999999, (event.clientY - bounds.top) / bounds.height));
    const x = Math.floor(rect.x + u * rect.w), y = Math.floor(rect.y + v * rect.h);
    const offset = y * state.row.size[0] + x, oldId = state.oldLabels[offset], newId = state.newLabels[offset];
    const name = id => id < 0 ? 'void' : byId.get(id).name;
    $('inspector').replaceChildren();
    for (const text of [`像素 (${x}, ${y})`, `原始：${oldId} · ${name(oldId)}`, `修改后：${newId} · ${name(newId)}`]) {
      const span = document.createElement('span'); span.textContent = text; $('inspector').append(span);
    }
    const badge = document.createElement('span'); badge.className = oldId !== newId ? 'changed-badge' : '';
    badge.textContent = oldId !== newId ? '类别已修改' : '未修改'; $('inspector').append(badge);
    $('inspector').dataset.pixel = `${x},${y}`;
    $('inspector').dataset.before = oldId; $('inspector').dataset.after = newId;
    document.querySelectorAll('.crosshair').forEach(c => {
      c.hidden = false; c.style.left = `${u * 100}%`; c.style.top = `${v * 100}%`;
    });
  }

  for (const canvas of canvases) {
    canvas.addEventListener('pointermove', event => {
      if (dragging && dragging.canvas === canvas && state.ready) {
        const bounds = canvas.getBoundingClientRect(), [width, height] = state.row.size;
        state.center = [dragging.center[0] - (event.clientX-dragging.x) / bounds.width * width / state.zoom,
                        dragging.center[1] - (event.clientY-dragging.y) / bounds.height * height / state.zoom];
        drawPanels();
      }
      inspect(event, canvas);
    });
    canvas.addEventListener('pointerdown', event => {
      if (!state.ready || event.button !== 0) return;
      dragging = {canvas, x: event.clientX, y: event.clientY, center: [...state.center]};
      canvas.classList.add('dragging'); canvas.setPointerCapture(event.pointerId);
    });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(type, () => {
      dragging = null; canvas.classList.remove('dragging');
    });
    canvas.addEventListener('pointerleave', () => document.querySelectorAll('.crosshair').forEach(c => { c.hidden = true; }));
  }

  $('split').addEventListener('change', () => { updateOptions(); filterSamples(); });
  $('category').addEventListener('change', () => { $('instance').value = 'all'; updateOptions(); filterSamples(); });
  $('instance').addEventListener('change', () => filterSamples());
  $('changedOnly').addEventListener('change', () => filterSamples(state.row?.id));
  $('search').addEventListener('input', () => filterSamples());
  $('reset').addEventListener('click', () => {
    $('split').value = 'all'; $('category').value = 'all'; $('instance').value = 'all';
    $('search').value = ''; $('changedOnly').checked = false;
    updateOptions(); filterSamples();
  });
  $('previous').addEventListener('click', () => showSample(state.index - 1));
  $('next').addEventListener('click', () => showSample(state.index + 1));
  $('pagePrev').addEventListener('click', () => { state.page--; renderList(); $('sampleList').scrollTop = 0; });
  $('pageNext').addEventListener('click', () => { state.page++; renderList(); $('sampleList').scrollTop = 0; });
  $('opacity').addEventListener('input', event => {
    state.opacity = Number(event.target.value) / 100;
    $('opacityValue').textContent = event.target.value + '%'; drawPanels();
  });
  function setMode(mode) {
    state.mode = mode;
    for (const [id, active] of [['overlayMode', mode === 'overlay'], ['labelMode', mode === 'labels']]) {
      $(id).classList.toggle('active', active); $(id).setAttribute('aria-pressed', String(active));
    }
    $('opacity').disabled = mode === 'labels'; drawPanels();
  }
  $('overlayMode').addEventListener('click', () => setMode('overlay'));
  $('labelMode').addEventListener('click', () => setMode('labels'));
  $('fitObject').addEventListener('click', () => setView('object'));
  $('fitImage').addEventListener('click', () => setView('image'));
  $('zoom').addEventListener('input', event => {
    if (!state.ready) return;
    state.zoom = Number(event.target.value);
    $('fitObject').classList.remove('active'); $('fitImage').classList.remove('active');
    updateZoom(); drawPanels();
  });
  document.addEventListener('keydown', event => {
    if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); showSample(state.index - 1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); showSample(state.index + 1); }
  });
  window.addEventListener('hashchange', () => openHash());
  function openHash() {
    const id = new URLSearchParams(location.hash.slice(1)).get('sample');
    if (!id || state.row?.id === id || !samples.some(r => r.id === id)) return false;
    $('split').value = 'all'; $('category').value = 'all'; $('instance').value = 'all';
    $('changedOnly').checked = false; $('search').value = '';
    updateOptions(); filterSamples(id); return true;
  }
  const summary = dataset.summary;
  for (const [value, title, extra] of [[summary.total, '总图像', false], [summary.test, '测试图', false],
                                     [summary.train, '训练示例', true], [summary.changed_images, '发生修改', false]]) {
    const cell = document.createElement('span'); cell.className = extra ? 'extra-stat' : '';
    const number = document.createElement('b'); number.textContent = fmt(value);
    cell.append(number, title); $('totals').append(cell);
  }
  updateOptions(); renderLegend();
  if (!openHash()) filterSamples();
})();
