"""
STATISTICAL domain-gap analysis (handcrafted features -> MMD headline).

Levels : scene (26 vs 26) | bbox (300 vs 300, 60/class) | mask (300 vs 300)
Pairs  : real-2D, real-3D, 2D-3D   (real is the target domain)
Metrics: MMD^2 (+perm p) [headline], energy distance, KID, proxy-A AUC,
         CORAL, hue-EMD companion.  Equal-N, bootstrapped CIs.

Caches feature matrices to domain_gap/cache/ so the latent stage can reuse
the exact same sampled instances.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import gap_common as gc
import gap_features as gf
import gap_distances as gd

OUT = Path(__file__).parent / "outputs"
CACHE = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

N_SCENE = 26
PER_CLASS = 60
HUE_CAP = 3000          # max hue pixels sampled per image/crop
B = 300                 # bootstrap resamples
N_PERM = 500
SEED = 0
DOMS = ["real", "2D", "3D"]
PAIRS = [("real", "2D"), ("real", "3D"), ("2D", "3D")]


def _hue_sample(hue, rng):
    if len(hue) > HUE_CAP:
        hue = hue[rng.choice(len(hue), HUE_CAP, replace=False)]
    return hue.astype(np.float32)


def extract():
    """Build/lo­ad feature matrices for every domain and level."""
    cache_f = CACHE / "handcrafted.npz"
    if cache_f.exists():
        print(f"[cache] loading {cache_f.name}")
        z = np.load(cache_f, allow_pickle=True)
        return {k: z[k].item() if z[k].dtype == object else z[k] for k in z}

    rng = np.random.RandomState(SEED)
    data = {}
    for dom in DOMS:
        t0 = time.time()
        splits = ["train", "val", "test"] if dom == "real" else ["train"]
        # ---- scene level ----
        scenes = gc.sample_scenes(dom, splits, None if dom == "real" else N_SCENE, seed=SEED)
        sfeat, shue = [], []
        for p in scenes:
            im = gc.load_scene(p)
            f, hue = gf.handcrafted(im)
            sfeat.append(f)
            shue.append(_hue_sample(hue, rng))
        # ---- instance levels (bbox + mask) share sampled instances ----
        inst = gc.sample_instances(
            dom, ["test"] if dom == "real" else ["train"],
            per_class=PER_CLASS, seed=SEED,
            per_scene_cap=None if dom == "real" else 4)
        by_img = defaultdict(list)
        for it in inst:
            by_img[it["image_path"]].append(it)
        bfeat, mfeat, labels, bhue, mhue = [], [], [], [], []
        for img_path, its in by_img.items():
            full = gc.load_full(img_path)
            for it in its:
                box, masked, fm = gc.crops_from(full, it["mask_path"])
                if box is None:
                    continue
                fb, hb = gf.handcrafted(box)
                fm2, hm = gf.handcrafted(masked, mask=fm)
                bfeat.append(fb); bhue.append(_hue_sample(hb, rng))
                mfeat.append(fm2); mhue.append(_hue_sample(hm, rng))
                labels.append(it["class"])
        data[dom] = dict(
            scene=np.array(sfeat), scene_hue=np.concatenate(shue),
            bbox=np.array(bfeat), bbox_hue=np.concatenate(bhue),
            mask=np.array(mfeat), mask_hue=np.concatenate(mhue),
            labels=np.array(labels))
        print(f"  {dom}: scene={data[dom]['scene'].shape} "
              f"bbox={data[dom]['bbox'].shape} mask={data[dom]['mask'].shape} "
              f"({time.time()-t0:.1f}s)")
    np.savez(cache_f, **{dom: data[dom] for dom in DOMS})
    print(f"[cache] saved {cache_f.name}")
    return data


def compare(X, Y, hue_x=None, hue_y=None):
    """Shared distances + the hue-EMD companion (statistical only)."""
    res = gd.compare(X, Y, n_perm=N_PERM, B=B, seed=SEED)
    if hue_x is not None:
        res["hue_emd"] = gd.hue_emd(hue_x, hue_y)
    return res


def main():
    data = extract()
    results = {"levels": {}, "per_class": {}}
    for level in ["scene", "bbox", "mask"]:
        hk = level + "_hue"
        results["levels"][level] = {}
        for a, b in PAIRS:
            r = compare(data[a][level], data[b][level],
                        data[a][hk], data[b][hk])
            results["levels"][level][f"{a}-{b}"] = r
            print(f"[{level:5s}] {a}-{b}: MMD={r['mmd']:.4f} "
                  f"(p={r['mmd_p']:.3f}, CI=[{r['mmd_ci'][0]:.4f},{r['mmd_ci'][1]:.4f}]) "
                  f"AUC={r['auc']:.3f} energy={r['energy']:.3f} "
                  f"KID={r['kid']:.4f} hueEMD={r.get('hue_emd',float('nan')):.2f}")
    # per-class at instance levels for the AP-correlation
    for level in ["bbox", "mask"]:
        results["per_class"][level] = {}
        for a, b in [("real", "2D"), ("real", "3D")]:
            results["per_class"][level][f"{a}-{b}"] = {}
            for c in gc.CLASSES:
                Xa = data[a][level][data[a]["labels"] == c]
                Xb = data[b][level][data[b]["labels"] == c]
                Xs, Ys = gd.standardize(Xa, Xb)
                mmd, p = gd.mmd_rbf(Xs, Ys, n_perm=0)
                auc, _ = gd.proxy_a_distance(Xs, Ys)
                results["per_class"][level][f"{a}-{b}"][c] = {
                    "mmd": mmd, "auc": auc}
            row = results["per_class"][level][f"{a}-{b}"]
            print(f"[per-class {level} {a}-{b}] " +
                  " ".join(f"{c}:{row[c]['mmd']:.3f}" for c in gc.CLASSES))
    with open(OUT / "statistical_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {OUT/'statistical_results.json'}")


if __name__ == "__main__":
    main()
