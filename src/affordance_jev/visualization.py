import colorsys
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size=18):
    path = Path(__file__).resolve().parent / "assets/DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def overlay(rgb, masks, colors=None, labels=None, fill_alpha=0.45):
    if not 0 <= fill_alpha <= 1:
        raise ValueError("fill_alpha must be between zero and one")
    canvas = np.array(rgb).copy()
    for i, mask in enumerate(masks):
        color = colors[i] if colors is not None else np.array(colorsys.hsv_to_rgb(((i+1)*.618) % 1, .72, .95)) * 255
        canvas[mask] = canvas[mask] * (1 - fill_alpha) + np.asarray(color) * fill_alpha
        edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        canvas[edge] = color
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        label = labels[i] if labels is not None else str(i+1)
        draw.text((x, y), label, font=font(max(12, min(22, image.width//25))), anchor="mm",
                  fill="white", stroke_width=2, stroke_fill="black")
    return image


def sheet(panels, path, columns=3, width=260, height=240):
    if not panels:
        panels = [("No candidates", Image.new("RGB", (width, height-30), "white"))]
    canvas = Image.new("RGB", (columns*width, ((len(panels)+columns-1)//columns)*height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(panels):
        x, y = (i % columns)*width, (i // columns)*height
        scale = min((width-8)/image.width, (height-35)/image.height)
        thumb = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x+(width-thumb.width)//2, y+30+(height-35-thumb.height)//2))
        draw.text((x+6, y+5), label, fill="black", font=font(14))
    canvas.save(path)


def report(out, status, timings=None):
    out = Path(out)
    groups = [
        ("1. RGB 与物体定位", ["rgb.png", "foreground/object_candidates.jpg", "foreground/object_overlay.png", "foreground/object_mask.png"]),
        ("2. 物体裁剪与 PartGLEE 输入", ["foreground/rgb_crop.png", "foreground/input_crop.png", "foreground/crop_mask.png"]),
        ("3. PartGLEE 原始部件候选（允许重叠）", ["raw/part_candidates.jpg"]),
        ("4. 过滤与互斥编号区域", ["regions/filtered_candidates.jpg", "regions/numbered_full.png", "regions/numbered_crop.png", "regions/region_sheet.jpg", "regions/unassigned_foreground.png"]),
        ("5. Valen 分类与按类别合并", ["result/classified_regions.png", "result/affordance_overlay.png", "result/class_sheet.jpg"]),
    ]
    body = [f"<h1>PartGLEE → Valen-4B</h1><p>状态：{html.escape(status)}</p>"]
    body.append("<p>编号区域逐像素互斥，每区域按配置选择一类；none 与未分配像素分开保存。模型标签和分数不是人工标注。</p>")
    for title, files in groups:
        body.append(f"<h2>{title}</h2><div class='grid'>")
        for name in files:
            if (out/name).is_file():
                body.append(f"<figure><a href='{name}'><img src='{name}' loading='lazy'></a><figcaption>{name}</figcaption></figure>")
        body.append("</div>")
    if (out/"decisions.json").is_file():
        data = json.loads((out/"decisions.json").read_text())
        names = [c["name"] for c in json.loads((out/"affordances_snapshot.json").read_text())["classes"]]
        body.append("<h2>逐区域类别概率</h2><table border='1' cellpadding='6' style='border-collapse:collapse'><tr><th>ID</th><th>模型类别</th><th>最终类别</th>"+"".join(f"<th>{n}</th>" for n in names)+"</tr>")
        for row in data["regions"]:
            body.append(f"<tr><td>{row['region_id']}</td><td>{html.escape(row['model_choice'])}</td><td>{html.escape(row['final_choice'])}</td>"+"".join(f"<td>{row['probabilities'][n]:.1%}</td>" for n in names)+"</tr>")
        body.append("</table><p>概率只针对当前配置的选项，不代表实测准确率。</p>")
    body.append("<h2>原始数据与计时</h2><ul>")
    for name in ["run.json", "job.json", "timings.json", "audit.json", "source_files_sha256.json", "prepare_runtime.json", "partglee_runtime.json", "checkpoint_loading.json", "config_snapshot.json", "affordances_snapshot.json", "foreground/metadata.json", "foreground/selection_trace.json", "raw/objects.json", "raw/parts.json", "regions/filter_trace.json", "candidates.json", "prepared_assets.json", "valen_request.json", "valen_response.json", "valen_runtime.json", "decisions.json", "result/masks.json", "logs/partglee.log", "logs/valen.log"]:
        if (out/name).is_file():
            body.append(f"<li><a href='{name}'>{name}</a></li>")
    body.append("</ul>")
    if (out/"valen_compiled").exists():
        tensor_link=" · <a href='valen_compiled/inputs.pt'>输入张量</a>" if (out/"valen_compiled/inputs.pt").is_file() else ""
        body.append("<p>Valen 实际编译输入：<a href='valen_compiled/prompt.txt'>完整编译 prompt</a> · <a href='valen_compiled/metadata.json'>张量尺寸与读取位置</a>"+tensor_link+"</p>")
    if timings:
        body.append("<pre>"+html.escape(json.dumps(timings, ensure_ascii=False, indent=2))+"</pre>")
    document = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>PartGLEE / Valen 分步检查</title><style>body{font-family:system-ui;margin:28px;background:white;color:#222}h1{font-size:25px}h2{font-size:19px;margin-top:30px}.grid{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;max-width:600px}img{max-width:100%;max-height:650px;border:1px solid #ddd}figcaption{font-size:13px}pre{white-space:pre-wrap}a{color:#1260a0}</style>"+"\n".join(body)+"</html>"
    (out/"index.html").write_text(document, encoding="utf-8")
