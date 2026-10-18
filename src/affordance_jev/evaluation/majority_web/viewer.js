"use strict";
(() => {
  const data = window.MAJORITY_DATA;
  if (!data) throw new Error("缺少 data.js，请复制完整导出目录。");
  const $ = id => document.getElementById(id);
  const pct = v => v == null ? "—" : (100 * v).toFixed(2) + "%";
  const rgbColor = color => `rgb(${color.join(",")})`;
  const cache = new Map();
  let filtered = data.samples.slice(), index = 0, token = 0, loaded = null, selectedRegion = 0;
  const metric = data.metrics;
  $("summary").textContent = `${metric.images.toLocaleString()} 张 RGB · ${metric.gt_present_queries.toLocaleString()} 个图像＋功能对　　mIoU ${pct(metric.foreground_mIoU)}　　gIoU ${pct(metric.gIoU_present_queries)}`;
  $("legend").innerHTML = data.classes.filter(c => c.id).map(c => `<span><i class="swatch" style="background:${rgbColor(c.color)}"></i>${c.id} ${c.name}</span>`).join("") + '<span class="muted">0 background / none（透明）</span>';
  const option = (value, text) => new Option(text, value);
  [...new Set(data.samples.map(r => r.object_category))].sort().forEach(v => $("category").add(option(v, v)));

  function updateInstances() {
    const current = $("instance").value;
    $("instance").replaceChildren(option("", "全部实例"));
    [...new Set(data.samples.filter(r => !$("category").value || r.object_category === $("category").value).map(r => r.object_instance))].sort().forEach(v => $("instance").add(option(v, v)));
    if ([...$("instance").options].some(o => o.value === current)) $("instance").value = current;
  }
  function applyFilters(keepCurrent = false) {
    const current = filtered[index]?.sample_id;
    const term = $("search").value.trim().toLowerCase();
    filtered = data.samples.filter(r => (!$("category").value || r.object_category === $("category").value) && (!$("instance").value || r.object_instance === $("instance").value) && (!term || r.sample_id.toLowerCase().includes(term)));
    $("sample").replaceChildren(...filtered.map(r => option(r.sample_id, r.sample_id)));
    index = keepCurrent ? Math.max(0, filtered.findIndex(r => r.sample_id === current)) : 0;
    show();
  }
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`图片读取失败：${url}`));
      image.src = url;
    });
  }
  async function imagesFor(row) {
    if (!cache.has(row.sample_id)) {
      const files = ["rgb.jpg", "candidates.png", "candidates_edges.png", "majority.png", "majority_edges.png", "gt.png", "gt_edges.png"];
      cache.set(row.sample_id, Promise.all(files.map(f => loadImage(row.path + f))));
      while (cache.size > 6) cache.delete(cache.keys().next().value);
    }
    return cache.get(row.sample_id);
  }
  function rectangle(row) {
    return $("crop").value === "full" ? [0, 0, ...row.size_wh] : $("crop").value === "source" ? row.source_crop_xyxy : row.crop_xyxy;
  }
  function draw() {
    if (!loaded) return;
    const {row, images} = loaded;
    const [x0, y0, x1, y1] = rectangle(row), width = x1 - x0, height = y1 - y0;
    const alpha = Number($("opacity").value) / 100;
    $("opacity-value").textContent = $("opacity").value + "%";
    ["rgb", "candidates", "majority", "gt"].forEach((name, p) => {
      const canvas = $(name), ctx = canvas.getContext("2d");
      const scale = Math.min((canvas.width - 14) / width, (canvas.height - 14) / height);
      const w = width * scale, h = height * scale, dx = (canvas.width - w) / 2, dy = (canvas.height - h) / 2;
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#f7f7f7"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "white"; ctx.fillRect(dx, dy, w, h);
      ctx.imageSmoothingEnabled = true;
      if (p === 0 || !$("mask-only").checked) ctx.drawImage(images[0], x0, y0, width, height, dx, dy, w, h);
      if (p > 0) {
        ctx.imageSmoothingEnabled = false;
        ctx.globalAlpha = $("mask-only").checked ? 1 : alpha;
        ctx.drawImage(images[2 * p - 1], x0, y0, width, height, dx, dy, w, h);
        ctx.globalAlpha = 1;
        if ($("edges").checked) ctx.drawImage(images[2 * p], x0, y0, width, height, dx, dy, w, h);
      }
      if (p === 1 && $("numbers").checked) {
        ctx.font = "bold 23px MajorityUI, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        row.regions.forEach(region => {
          const [x, y] = region.anchor_xy;
          if (x < x0 || x >= x1 || y < y0 || y >= y1) return;
          const px = dx + (x - x0) * scale, py = dy + (y - y0) * scale;
          ctx.beginPath(); ctx.arc(px, py, region.id === selectedRegion ? 23 : 17, 0, Math.PI * 2);
          ctx.fillStyle = region.id === selectedRegion ? "#fbdb6d" : "rgba(255,255,255,.9)"; ctx.fill();
          ctx.lineWidth = region.id === selectedRegion ? 4 : 2; ctx.strokeStyle = "#30363c"; ctx.stroke();
          ctx.fillStyle = "#20252c"; ctx.fillText(String(region.id), px, py + 1);
        });
      }
    });
  }
  function updateDetails(row) {
    $("sample-title").textContent = row.sample_id;
    $("sample-metrics").textContent = `${row.size_wh.join(" × ")} 像素 · ${row.regions.length} 个固定候选 · 本图平均 IoU ${pct(row.mean_present_iou)} · 候选未覆盖的 GT 功能像素 ${row.unassigned_gt_foreground_pixels.toLocaleString()}` + (row.background_majority_regions.length ? ` · 背景多数候选 ${row.background_majority_regions.join("、")}` : "") + (row.void_pixels ? ` · void ${row.void_pixels} 像素` : "");
    $("query-metrics").innerHTML = row.queries.map(q => `<span><i class="swatch" style="background:${rgbColor(data.classes[q.class_id].color)}"></i>${q.class} ${pct(q.iou)}</span>`).join("");
    $("regions").innerHTML = row.regions.map(r => `<tr data-region="${r.id}" tabindex="0"><td><span class="region-id"><i class="swatch" style="background:${rgbColor(data.region_colors[r.id])}"></i>${r.id}</span></td><td>${r.original_class}</td><td class="${r.original_class !== r.majority_class ? "changed" : ""}">${r.majority_class}</td><td>${pct(r.purity)}</td><td>${r.valid_pixels.toLocaleString()}</td><td>${r.gt_counts.map((n, c) => n ? `<span class="count">${c ? data.classes[c].name : "background"} <b>${n.toLocaleString()}</b></span>` : "").join("") || "无有效 GT 像素"}</td></tr>`).join("");
    document.querySelectorAll("#regions tr").forEach(tr => {
      const activate = () => { selectedRegion = selectedRegion === Number(tr.dataset.region) ? 0 : Number(tr.dataset.region); document.querySelectorAll("#regions tr").forEach(r => r.classList.toggle("selected", Number(r.dataset.region) === selectedRegion)); draw(); };
      tr.addEventListener("click", activate);
      tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") {e.preventDefault(); activate();} });
    });
    $("download-map").href = row.class_map_data_url;
    $("download-map").download = row.sample_id.replace("/", "_") + "_class_map.png";
    $("sample-json").href = row.path + "regions.json";
  }
  async function show() {
    const request = ++token;
    selectedRegion = 0;
    $("prev").disabled = index <= 0;
    $("next").disabled = index >= filtered.length - 1;
    $("counter").textContent = filtered.length ? `${index + 1} / ${filtered.length.toLocaleString()}` : "0 / 0";
    document.body.classList.toggle("empty", !filtered.length);
    $("status").className = "";
    if (!filtered.length) {
      loaded = null; $("status").textContent = "没有符合筛选条件的样本。"; $("sample-title").textContent = ""; $("sample-metrics").textContent = ""; $("query-metrics").replaceChildren(); $("regions").replaceChildren();
      ["rgb", "candidates", "majority", "gt"].forEach(id => $(id).getContext("2d").clearRect(0, 0, 800, 580));
      $("download-map").removeAttribute("href"); $("sample-json").removeAttribute("href");
      return;
    }
    const row = filtered[index]; $("sample").value = row.sample_id;
    $("status").textContent = "正在读取 " + row.sample_id;
    try {
      const images = await imagesFor(row);
      if (request !== token) return;
      loaded = {row, images}; updateDetails(row); draw();
      $("status").textContent = "原尺寸数据已载入；所有面板使用相同坐标和裁剪。";
      history.replaceState(null, "", "#" + encodeURIComponent(row.sample_id));
      document.body.dataset.sampleId = row.sample_id;
      document.body.dataset.ready = "true";
    } catch (error) { $("status").textContent = error.message; $("status").className = "error"; console.error(error); }
  }
  $("category").addEventListener("change", () => { $("instance").value = ""; updateInstances(); applyFilters(); });
  $("instance").addEventListener("change", () => applyFilters());
  $("search").addEventListener("input", () => applyFilters());
  $("sample").addEventListener("change", () => { index = filtered.findIndex(r => r.sample_id === $("sample").value); show(); });
  $("prev").addEventListener("click", () => { if (index > 0) {--index; show();} });
  $("next").addEventListener("click", () => { if (index < filtered.length - 1) {++index; show();} });
  ["crop", "opacity", "edges", "numbers", "mask-only"].forEach(id => $(id).addEventListener("input", draw));
  document.addEventListener("keydown", event => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    if (event.key === "ArrowRight") $("next").click();
    if (event.key === "ArrowLeft") $("prev").click();
  });
  updateInstances(); applyFilters();
  const initial = decodeURIComponent(location.hash.slice(1));
  const initialIndex = filtered.findIndex(r => r.sample_id === initial);
  if (initialIndex > 0) {index = initialIndex; show();}
})();
