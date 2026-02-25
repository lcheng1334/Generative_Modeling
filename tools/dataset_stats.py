"""
dataset_stats.py - 私有电感 AOI 数据集完整统计
输出: 每类缺陷 x 每工位 x 每姿态的样本数量表
"""
import sys
sys.path.insert(0, ".")

from collections import defaultdict
from src.datasets.defect_dataset import DefectDataset, DEFECT_TYPES, CAM_IDS

DATA_ROOT = r"E:\code\dataset\Generative_Modeling\data\datasets"


def print_table(title, rows, headers):
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f" {{:<{w-1}}}" for w in col_w) + "|"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def main():
    # ── NG 统计 ──────────────────────────────────────
    ng_ds = DefectDataset(DATA_ROOT, mode="ng")
    ok_ds = DefectDataset(DATA_ROOT, mode="ok")

    # 按 (defect, cam, pose) 统计
    ng_counter = defaultdict(int)
    for s in ng_ds.samples:
        # s['defect'], cam, pose
        import re
        info = re.match(r".*/(.*?)/(.*?)/", s["path"].replace("\\", "/"))
        ng_counter[(s["defect"], s["cam"], s["pose"])] += 1

    # 按 (cam, pose) 统计 OK
    ok_counter = defaultdict(int)
    for s in ok_ds.samples:
        ok_counter[(s["cam"], s["pose"])] += 1

    # ── 打印 NG 缺陷明细表 ──────────────────────────
    rows = []
    grand_total = 0
    for defect in DEFECT_TYPES:
        p0 = sum(ng_counter[(defect, c, "p0")] for c in CAM_IDS)
        p90 = sum(ng_counter[(defect, c, "p90")] for c in CAM_IDS)
        total = p0 + p90
        grand_total += total
        # 有效工位
        valid_cams = sorted({c for c in CAM_IDS
                             if ng_counter[(defect, c, "p0")] + ng_counter[(defect, c, "p90")] > 0})
        cam_str = " ".join(f"Cam{c}" for c in valid_cams)
        rows.append([defect, cam_str, p0, p90, total])

    print_table(
        "NG Dataset Summary",
        rows,
        ["Defect", "Valid Cams", "p0", "p90", "Total"]
    )
    print(f"  Grand Total NG: {grand_total}")

    # ── 打印 OK 明细表 ──────────────────────────────
    ok_rows = []
    ok_total = 0
    for cam in CAM_IDS:
        p0  = ok_counter[(cam, "p0")]
        p90 = ok_counter[(cam, "p90")]
        ok_rows.append([f"Cam{cam}", p0, p90, p0 + p90])
        ok_total += p0 + p90

    print_table(
        "OK Dataset Summary",
        ok_rows,
        ["Camera", "p0", "p90", "Total"]
    )
    print(f"  Grand Total OK: {ok_total}")

    # ── 每工位 NG 详细分布 ──────────────────────────
    cam_rows = []
    for cam in CAM_IDS:
        for defect in DEFECT_TYPES:
            p0  = ng_counter[(defect, cam, "p0")]
            p90 = ng_counter[(defect, cam, "p90")]
            if p0 + p90 > 0:
                cam_rows.append([f"Cam{cam}", defect, p0, p90, p0 + p90])

    print_table(
        "NG Per-Camera Breakdown",
        cam_rows,
        ["Camera", "Defect", "p0", "p90", "Total"]
    )

    print(f"\n[DONE] Total samples: {grand_total} NG + {ok_total} OK = {grand_total + ok_total}")


if __name__ == "__main__":
    main()
