"""Figures and an offline gallery for a completed cached-output diagnosis."""
import argparse
import html
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from fontTools import subset
from fontTools.ttLib import TTFont

from ..common import read, read_jsonl, write, padded_bbox
from ..datasets.umd import load_annotation
from ..visualization import font, overlay, sheet


CASES = [
    ("shovel_01/00000082", "support", "功能定义混淆", "GT 为 support 的铲面被判为 scoop；候选区域本身已基本覆盖铲面。"),
    ("ladle_01/00000292", "contain", "UMD 标签与常识角色不同", "原始 UMD 将勺头标为 contain，当前判断多为 scoop；本报告保持原始 GT。"),
    ("cup_01/00000001", "wrap-grasp", "内外表面类别颠倒", "杯身外侧多个候选被判为 contain，内外表面的对应关系出错。"),
    ("scoop_02/00000013", "scoop", "候选可用，但类别不对", "现有候选组合可接近 GT，实际 scoop 预测几乎为空。"),
    ("tenderizer_01/00000040", "pound", "目标选成底板，后续过滤失效", "大面积蓝色底板赢得物体排序；最小连通区域阈值升至 2634 像素，真实敲击部件被过滤。"),
    ("saw_03/00000151", "grasp", "单个候选混合了不同 GT 区域", "整块锯柄候选同时包含 GT grasp 和 GT 背景；一个候选只能赋一类，无法精确表达 GT。"),
    ("tenderizer_02/00000001", "pound", "物体 mask 漏掉功能像素", "目标 mask 对全部 GT 功能像素的覆盖仅 71.03%，后续阶段无法恢复已裁掉的区域。"),
    ("trowel_01/00000001", "scoop", "正常对照", "展示同一流程能够较好完成的工具部件，便于对照失败模式。"),
]


def bundle_font(root, document):
    """Subset the existing local CJK font for the new report's actual text."""
    project = Path(__file__).resolve().parents[4]
    source = project / "outputs/wps_experiments_20261013/fonts/NotoSansCJKsc-Regular.otf"
    font_path = root / "fonts"
    font_path.mkdir(exist_ok=True)
    ft = TTFont(source)
    sub = subset.Subsetter()
    sub.populate(text=document + "".join(chr(i) for i in range(32, 127)))
    sub.subset(ft)
    for record in ft["name"].names:
        if record.nameID in {1, 4, 6, 16}:
            record.string = "UMDDiagnosisUI".encode(record.getEncoding())
    if "CFF " in ft:
        ft["CFF "].cff.fontNames = ["UMDDiagnosisUI"]
        ft["CFF "].cff.topDictIndex[0].FullName = "UMDDiagnosisUI"
        ft["CFF "].cff.topDictIndex[0].FamilyName = "UMDDiagnosisUI"
    ft.flavor = "woff"
    ft.save(font_path / "umd-ui.woff")
    license_path = Path(__file__).resolve().parents[1] / "label_viewer/web/fonts/OFL.txt"
    shutil.copyfile(license_path, font_path / "OFL.txt")
    (font_path / "README.txt").write_text("UMDDiagnosisUI is a renamed report-specific subset of Noto Sans CJK SC, under the bundled SIL Open Font License.\n")


def make_report(analysis):
    root = Path(analysis).resolve()
    summary = read(root / "summary.json")
    images = {r["sample_id"]: r for r in read_jsonl(root / "per_image.jsonl")}
    queries = {(r["sample_id"], r["class"]): r for r in read_jsonl(root / "per_query.jsonl")}
    regions = read_jsonl(root / "per_region.jsonl")
    absent_counts = {}
    for region in regions:
        name = region["prediction"]
        if name == "none":
            continue
        count = absent_counts.setdefault(name, {"pred_pixels": 0, "absent_class_pred_pixels": 0, "regions": 0, "absent_class_regions": 0})
        count["pred_pixels"] += region["valid_pixels"]
        count["regions"] += 1
        if (region["sample_id"], name) not in queries:
            count["absent_class_pred_pixels"] += region["valid_pixels"]
            count["absent_class_regions"] += 1
    write(root / "absent_class_predictions.json", absent_counts)
    manifest = {r["sample_id"]: r for r in read_jsonl(summary["manifest"])}
    source = Path(summary["predictions"])
    definitions = read(source / "affordances_snapshot.json")
    classes = sorted([c for c in definitions["classes"] if c["id"]], key=lambda c: c["id"])
    visualizations = root / "visualizations"
    visualizations.mkdir(exist_ok=True)

    names = [c["name"] for c in classes]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    a = [100 * summary["actual"]["per_class"][n]["iou"] for n in names]
    b = [100 * summary["gt_majority_relabel_diagnostic"]["per_class"][n]["iou"] for n in names]
    ax.bar(x - .19, a, width=.38, color="#477ea9", label="Actual prediction")
    ax.bar(x + .19, b, width=.38, color="#a5bdca", label="GT-majority relabel (diagnostic)")
    for xx, aa, bb in zip(x, a, b):
        ax.text(xx - .19, aa + 1, f"{aa:.1f}", ha="center", fontsize=9)
        ax.text(xx + .19, bb + 1, f"{bb:.1f}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 106), ylabel="Dataset class IoU (%)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "per_function_iou.png", dpi=170)
    plt.close(fig)

    error_names = ["Foreground misses", "Unassigned foreground", "Mixed candidate pixels", "Recoverable by relabeling"]
    values = [100 * summary["pixel_error_decomposition"][k] / summary["total_pixel_errors"] for k in
              ["foreground_missed_gt", "partition_unassigned_gt", "mixed_region_unavoidable", "decision_excess_over_majority"]]
    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    ax.barh(error_names, values, color=["#bac5cc", "#bac5cc", "#7f9cb0", "#477ea9"])
    for i, value in enumerate(values):
        ax.text(value + .7, i, f"{value:.2f}%", va="center")
    ax.invert_yaxis()
    ax.set(xlim=(0, 76), xlabel="Share of incorrectly labeled pixels (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "pixel_error_breakdown.png", dpi=170)
    plt.close(fig)

    cards, case_records = [], []
    for sid, focus, title, note in CASES:
        row, im = manifest[sid], images[sid]
        sample = source / "samples" / sid
        rgb = Image.open(row["rgb"]).convert("RGB")
        gt, _ = load_annotation(row["annotation"], tuple(row["size_wh"][::-1]))
        pred = np.asarray(Image.open(sample / "result/class_map.png"))
        labels = np.asarray(Image.open(sample / "regions/label_map.png"))
        majority = np.array(im["majority_labels"], dtype=np.uint8)[labels]
        fg = np.asarray(Image.open(sample / "foreground/object_mask.png")) > 0
        crop = padded_bbox((gt > 0) | fg, .12)
        def painted(class_map):
            return overlay(rgb, [class_map == c["id"] for c in classes],
                           colors=[c["color"] for c in classes], labels=["" for _ in classes]).crop(crop)
        numbered = Image.open(sample / "regions/numbered_full.png").convert("RGB").crop(crop)
        relative = "visualizations/" + sid.replace("/", "_") + ".jpg"
        sheet([("RGB", rgb.crop(crop)), ("Original GT", painted(gt)),
               ("Fixed candidates", numbered), ("Valen prediction", painted(pred)),
               ("GT-majority (diagnostic)", painted(majority))], root / relative, columns=5, width=330, height=320)
        panel = Image.open(root / relative).convert("RGB")
        canvas = Image.new("RGB", (panel.width, panel.height + 44), "white")
        canvas.paste(panel)
        draw = ImageDraw.Draw(canvas)
        for i, cls in enumerate(classes):
            px = 18 + i * 225
            draw.rectangle((px, panel.height + 13, px + 14, panel.height + 27), fill=tuple(cls["color"]))
            draw.text((px + 23, panel.height + 10), cls["name"], fill="black", font=font(15))
        canvas.save(root / relative, quality=94)
        q = queries[sid, focus]
        case_records.append({"sample_id": sid, "focus": focus, "title": title, "note": note, "query": q,
                             "figure": relative, "selection": "Diagnostic mechanism example, not random sampling"})
        rows = [r for r in regions if r["sample_id"] == sid]
        table = "".join(
            f"<tr><td>{r['region_id']}</td><td>{r['gt_majority_class']}</td><td>{r['purity']:.1%}</td>"
            f"<td>{r['prediction']}</td><td>{r['prediction_probability']:.1%}</td></tr>" for r in rows)
        original = "../umd_novel_test/samples/" + sid + "/index.html"
        cards.append(
            f"<section><h2>{html.escape(title)} <span>{sid}</span></h2><p>{html.escape(note)}</p>"
            f"<p>{focus} IoU：实际 <b>{q['actual_iou']:.2%}</b>；现有候选最佳组合 <b>{q['best_union_iou']:.2%}</b>（使用 GT 的诊断值）。</p>"
            f"<a href='{relative}'><img src='{relative}' loading='lazy'></a>"
            f"<details><summary>逐区域判断与原始中间结果</summary><table><tr><th>编号</th><th>GT 多数类别</th><th>GT 纯度</th><th>Valen 类别</th><th>模型概率</th></tr>{table}</table>"
            f"<p><a href='{original}'>打开原始完整推理流程</a>（模型概率不是准确率）</p></details></section>"
        )
    write(root / "cases.json", case_records)

    # Inspect every extreme foreground mismatch, rather than a single selected frame.
    extreme = sorted((r for r in images.values() if r["foreground_precision_against_annotated_pixels"] < .05), key=lambda r: r["sample_id"])
    sheet([(r["sample_id"], Image.open(source / "samples" / r["sample_id"] / "foreground/object_overlay.png").convert("RGB"))
           for r in extreme], root / "extreme_foreground_cases.jpg", columns=5, width=320, height=270)
    write(root / "extreme_foreground_cases.json", {"selection": "selected foreground precision against GT-positive pixels < 5%", "samples": extreme})
    headline = (
        "<h1>PartGLEE＋Valen：UMD 误差定位</h1>"
        "<p>2,661 张人工标注测试 RGB；原始 dominant GT。只分析已有输出，未修改预测或运行模型。</p>"
        "<table><tr><th>设置</th><th>mIoU</th><th>gIoU</th></tr>"
        f"<tr><td>实际预测</td><td>{summary['actual']['foreground_mIoU']:.2%}</td><td>{summary['queries']['actual_iou']:.2%}</td></tr>"
        f"<tr><td>固定候选＋GT 多数类别（诊断）</td><td>{summary['gt_majority_relabel_diagnostic']['foreground_mIoU']:.2%}</td><td>{summary['queries']['majority_iou']:.2%}</td></tr>"
        f"<tr><td>各功能独立选择最佳候选组合（诊断上限）</td><td>不适用</td><td>{summary['queries']['best_union_iou']:.2%}</td></tr></table>"
        "<p>GT 多数类别最大化像素正确数，并非 mIoU 最优；各功能独立最优组合可能冲突。以上诊断值不能当作模型成绩。</p>"
        "<p>可重新赋类修复的错误包含目标选错和功能定义不一致，不能全部归因于 Valen 模型本身。下方案例用于解释失败机制，不是随机抽样。</p>"
        "<div class='charts'><img src='per_function_iou.png'><img src='pixel_error_breakdown.png'></div>"
        "<p><a href='summary.json'>完整统计</a> · <a href='per_query.jsonl'>全部查询诊断</a> · <a href='per_region.jsonl'>全部区域判断</a> · "
        "<a href='extreme_foreground_cases.jpg'>全部 10 张极端前景异常</a></p>"
    )
    document = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>UMD 误差定位</title><style>@font-face{font-family:UMD;src:url(fonts/umd-ui.woff)}"
        "body{font-family:UMD,system-ui,sans-serif;max-width:1600px;margin:28px auto;padding:0 18px;color:#222;background:white;line-height:1.6}"
        "h1{font-size:26px}h2{font-size:20px}h2 span{font-size:15px;font-weight:normal;margin-left:14px}"
        "table{border-collapse:collapse}td,th{padding:8px 18px;text-align:left;border-bottom:1px solid #ddd}"
        "img{max-width:100%;height:auto}.charts{display:flex;align-items:center;gap:20px}.charts img{width:49%}"
        "section{border-top:1px solid #ddd;margin-top:32px;padding-top:12px}a{color:#185681}summary{cursor:pointer}"
        "@media(max-width:800px){.charts{display:block}.charts img{width:100%}body{margin:16px auto}td,th{padding:6px}}</style>"
        + headline + "".join(cards) + "</html>\n")
    bundle_font(root, document)
    (root / "index.html").write_text(document, encoding="utf-8")
    print(root / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    make_report(**vars(parser.parse_args()))
