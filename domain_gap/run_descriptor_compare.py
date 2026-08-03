"""
Robustness check: compact (20-d) vs extended (36-d, +GLCM/LBP) descriptor.

Question: does adding the canonical texture descriptors (Haralick/GLCM, LBP)
change any CONCLUSION of the statistical domain-gap analysis? Absolute
MMD values are not comparable across descriptors of different dimensionality,
so the comparison is on the scale-free findings:

  F2  mask-level ordering + ratio   (is the 3D fragment gap larger than 2D's?)
  F3  level decomposition           (does removing the belt collapse the 2D gap
                                     more than the 3D gap?)
  F4  per-class gap-vs-AP Spearman  (does the gap predict per-class AP?)

If the findings agree, the compact descriptor is kept for the paper.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import gap_common as gc
import gap_features as gf
import gap_distances as gd

OUT = Path(__file__).parent / "outputs"
CACHE = Path(__file__).parent / "cache"
N_SCENE, PER_CLASS, SEED = 26, 60, 0
DOMS = ["real", "2D", "3D"]
LEVELS = ["scene", "bbox", "mask"]
AP = {  # paper Table 4: model trained purely on the measured dataset
    "2D": {"AAC": 0.669, "Ceramics": 0.944, "Mortar": 0.583, "Stones": 0.629, "Tiles": 0.912},
    "3D": {"AAC": 0.228, "Ceramics": 0.820, "Mortar": 0.414, "Stones": 0.027, "Tiles": 0.609},
}


def extract_ext():
    f = CACHE / "handcrafted_ext.npz"
    if f.exists():
        print(f"[cache] {f.name}")
        z = np.load(f, allow_pickle=True)
        return {k: z[k].item() for k in z}
    data = {}
    for dom in DOMS:
        t0 = time.time()
        splits = ["train", "val", "test"] if dom == "real" else ["train"]
        scenes = gc.sample_scenes(dom, splits, None if dom == "real" else N_SCENE, seed=SEED)
        sfeat = [gf.handcrafted_ext(gc.load_scene(p))[0] for p in scenes]
        inst = gc.sample_instances(dom, ["test"] if dom == "real" else ["train"],
                                   per_class=PER_CLASS, seed=SEED,
                                   per_scene_cap=None if dom == "real" else 4)
        by_img = defaultdict(list)
        for it in inst:
            by_img[it["image_path"]].append(it)
        bfeat, mfeat, labels = [], [], []
        for img_path, its in by_img.items():
            full = gc.load_full(img_path)
            for it in its:
                box, masked, fm = gc.crops_from(full, it["mask_path"])
                if box is None:
                    continue
                bfeat.append(gf.handcrafted_ext(box)[0])
                mfeat.append(gf.handcrafted_ext(masked, mask=fm)[0])
                labels.append(it["class"])
        data[dom] = dict(scene=np.array(sfeat), bbox=np.array(bfeat),
                         mask=np.array(mfeat), labels=np.array(labels))
        print(f"  {dom}: dim={data[dom]['scene'].shape[1]} ({time.time()-t0:.0f}s)")
    np.savez(f, **data)
    return data


def findings(get_mmd, get_pc):
    """Return the three scale-free findings for one descriptor."""
    m = {lv: {p: get_mmd(lv, p) for p in ["real-2D", "real-3D"]} for lv in LEVELS}
    f2_ratio = m["mask"]["real-3D"] / m["mask"]["real-2D"]
    f3_drop2d = (m["scene"]["real-2D"] - m["mask"]["real-2D"]) / m["scene"]["real-2D"]
    f3_drop3d = (m["scene"]["real-3D"] - m["mask"]["real-3D"]) / m["scene"]["real-3D"]
    f4 = {}
    for model in ["2D", "3D"]:
        pc = get_pc(model)
        gaps = [pc[c] for c in gc.CLASSES]
        aps = [AP[model][c] for c in gc.CLASSES]
        f4[model] = spearmanr(gaps, aps)[0]
    return dict(mmd=m, f2_ratio=f2_ratio, f3_drop2d=f3_drop2d,
                f3_drop3d=f3_drop3d, f4=f4)


def main():
    # --- current (compact) descriptor: reuse the committed results ---
    cur = json.loads((OUT / "statistical_results.json").read_text())
    compact = findings(
        lambda lv, p: cur["levels"][lv][p]["mmd"],
        lambda model: {c: cur["per_class"]["mask"][f"real-{model}"][c]["mmd"]
                       for c in gc.CLASSES})

    # --- extended descriptor ---
    data = extract_ext()
    ext_lv, ext_pc = {}, {}
    for lv in LEVELS:
        ext_lv[lv] = {}
        for a, b in [("real", "2D"), ("real", "3D")]:
            r = gd.compare(data[a][lv], data[b][lv], n_perm=200, B=100, seed=SEED)
            ext_lv[lv][f"{a}-{b}"] = r["mmd"]
    for model in ["2D", "3D"]:
        ext_pc[model] = {}
        for c in gc.CLASSES:
            Xa = data["real"]["mask"][data["real"]["labels"] == c]
            Xb = data[model]["mask"][data[model]["labels"] == c]
            Xs, Ys = gd.standardize(Xa, Xb)
            ext_pc[model][c] = gd.mmd_rbf(Xs, Ys, n_perm=0)[0]
    extended = findings(lambda lv, p: ext_lv[lv][p], lambda model: ext_pc[model])

    # --- report ---
    print("\n" + "=" * 72)
    dim = data["real"]["mask"].shape[1]
    print(f"{'':34s}{'compact (20-d)':>18s}{f'extended ({dim}-d)':>19s}")
    print("-" * 72)
    print("MMD^2 by level (absolute values are NOT comparable across descriptors)")
    for lv in LEVELS:
        for p in ["real-2D", "real-3D"]:
            print(f"  {lv+' '+p:<32s}{compact['mmd'][lv][p]:>18.3f}{extended['mmd'][lv][p]:>19.3f}")
    print("-" * 72)
    print("FINDINGS (scale-free -- these are what must agree)")
    print(f"  {'F2 mask ratio 3D/2D  (>1 = 3D worse)':<32s}"
          f"{compact['f2_ratio']:>18.2f}{extended['f2_ratio']:>19.2f}")
    print(f"  {'F3 belt removal: 2D gap drop':<32s}"
          f"{compact['f3_drop2d']*100:>17.0f}%{extended['f3_drop2d']*100:>18.0f}%")
    print(f"  {'F3 belt removal: 3D gap drop':<32s}"
          f"{compact['f3_drop3d']*100:>17.0f}%{extended['f3_drop3d']*100:>18.0f}%")
    for model in ["2D", "3D"]:
        print(f"  {f'F4 Spearman rho vs {model} AP':<32s}"
              f"{compact['f4'][model]:>+18.2f}{extended['f4'][model]:>+19.2f}")
    print("=" * 72)

    # --- verdict ---
    same_f2 = (compact["f2_ratio"] > 1) == (extended["f2_ratio"] > 1)
    same_f3 = (compact["f3_drop2d"] > compact["f3_drop3d"]) == \
              (extended["f3_drop2d"] > extended["f3_drop3d"])
    same_f4 = all(abs(compact["f4"][m]) < 0.5 and abs(extended["f4"][m]) < 0.5
                  or np.sign(compact["f4"][m]) == np.sign(extended["f4"][m])
                  for m in ["2D", "3D"])
    print(f"F2 ordering agrees : {same_f2}")
    print(f"F3 pattern agrees  : {same_f3}")
    # F4 (per-class Spearman) was cut from the paper; verdict rests on F2/F3.
    print(f"F4 (Spearman) agrees: {same_f4}  [informational; cut from paper]")
    print("\nVERDICT:", "paper conclusions (F2, F3) unchanged -> keep the compact descriptor"
          if (same_f2 and same_f3)
          else "findings DIFFER -> the extended descriptor changes a conclusion")
    json.dump({"compact": compact, "extended": extended},
              open(OUT / "descriptor_comparison.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
