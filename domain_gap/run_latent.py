"""
LATENT-SPACE domain-gap analysis (neutral encoders -> same distances).

Encoders: DINOv2 ViT-S/14 and COCO-init YOLOv11s backbone (both untrained on
these domains). Same sampled scenes/instances as the statistical stage
(deterministic seed), embedded at scene / bbox / mask level, then compared
with the shared distances (MMD headline + energy/KID/AUC/CORAL), equal-N
bootstrapped, plus per-class at the instance levels for the AP correlation.

Run with the cert bundle set so the one-time DINOv2 download works:
  SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())") \
  REQUESTS_CA_BUNDLE=$SSL_CERT_FILE python run_latent.py
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import gap_common as gc
import gap_encoders as ge
import gap_distances as gd

OUT = Path(__file__).parent / "outputs"
CACHE = Path(__file__).parent / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

YOLO_W = "/Users/tomas/Desktop/PycharmProjects/AICycle-DS/yolo11s.pt"
N_SCENE = 26
PER_CLASS = 60
B = 300
N_PERM = 500
SEED = 0
DOMS = ["real", "2D", "3D"]
PAIRS = [("real", "2D"), ("real", "3D"), ("2D", "3D")]


def collect_images():
    """Same deterministic samples as the statistical stage."""
    imgs = {d: {} for d in DOMS}
    for dom in DOMS:
        splits = ["train", "val", "test"] if dom == "real" else ["train"]
        scenes = gc.sample_scenes(dom, splits, None if dom == "real" else N_SCENE, seed=SEED)
        imgs[dom]["scene"] = [gc.load_scene(p) for p in scenes]
        inst = gc.sample_instances(
            dom, ["test"] if dom == "real" else ["train"],
            per_class=PER_CLASS, seed=SEED,
            per_scene_cap=None if dom == "real" else 4)
        by_img = defaultdict(list)
        for it in inst:
            by_img[it["image_path"]].append(it)
        box, mask, labels = [], [], []
        for img_path, its in by_img.items():
            full = gc.load_full(img_path)
            for it in its:
                b, m, fm = gc.crops_from(full, it["mask_path"])
                if b is None:
                    continue
                box.append(b); mask.append(m); labels.append(it["class"])
        imgs[dom]["bbox"] = box
        imgs[dom]["mask"] = mask
        imgs[dom]["labels"] = np.array(labels)
    return imgs


def build_embeddings():
    cache_f = CACHE / "embeddings.npz"
    if cache_f.exists():
        print(f"[cache] loading {cache_f.name}")
        z = np.load(cache_f, allow_pickle=True)
        return {k: z[k].item() for k in z}

    imgs = collect_images()
    encoders = {"dino": ge.DinoV2Encoder(), "yolo": ge.YoloBackboneEncoder(YOLO_W)}
    print(f"[device] {ge.pick_device()}")
    emb = {enc: {d: {} for d in DOMS} for enc in encoders}
    for enc_name, enc in encoders.items():
        for dom in DOMS:
            t0 = time.time()
            for level in ["scene", "bbox", "mask"]:
                emb[enc_name][dom][level] = enc.embed(imgs[dom][level])
            emb[enc_name][dom]["labels"] = imgs[dom]["labels"]
            print(f"  {enc_name}/{dom}: "
                  f"scene={emb[enc_name][dom]['scene'].shape} "
                  f"bbox={emb[enc_name][dom]['bbox'].shape} "
                  f"({time.time()-t0:.1f}s)")
    np.savez(cache_f, **{enc: emb[enc] for enc in encoders})
    print(f"[cache] saved {cache_f.name}")
    return emb


def main():
    emb = build_embeddings()
    results = {}
    for enc in emb:
        results[enc] = {"levels": {}, "per_class": {}}
        for level in ["scene", "bbox", "mask"]:
            results[enc]["levels"][level] = {}
            for a, b in PAIRS:
                r = gd.compare(emb[enc][a][level], emb[enc][b][level],
                               n_perm=N_PERM, B=B, seed=SEED)
                results[enc]["levels"][level][f"{a}-{b}"] = r
                print(f"[{enc}|{level:5s}] {a}-{b}: MMD={r['mmd']:.4f} "
                      f"(p={r['mmd_p']:.3f} CI=[{r['mmd_ci'][0]:.4f},{r['mmd_ci'][1]:.4f}]) "
                      f"AUC={r['auc']:.3f} KID={r['kid']:.4f}")
        for level in ["bbox", "mask"]:
            results[enc]["per_class"][level] = {}
            for a, b in [("real", "2D"), ("real", "3D")]:
                results[enc]["per_class"][level][f"{a}-{b}"] = {}
                for c in gc.CLASSES:
                    Xa = emb[enc][a][level][emb[enc][a]["labels"] == c]
                    Xb = emb[enc][b][level][emb[enc][b]["labels"] == c]
                    Xs, Ys = gd.standardize(Xa, Xb)
                    mmd, _ = gd.mmd_rbf(Xs, Ys, n_perm=0)
                    auc, _ = gd.proxy_a_distance(Xs, Ys)
                    results[enc]["per_class"][level][f"{a}-{b}"][c] = {
                        "mmd": mmd, "auc": auc}
                row = results[enc]["per_class"][level][f"{a}-{b}"]
                print(f"[{enc}|per-class {level} {a}-{b}] " +
                      " ".join(f"{c}:{row[c]['mmd']:.3f}" for c in gc.CLASSES))
    with open(OUT / "latent_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {OUT/'latent_results.json'}")


if __name__ == "__main__":
    main()
