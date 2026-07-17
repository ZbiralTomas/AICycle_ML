"""
Neutral latent encoders for the domain-gap analysis.

Two encoders that were NOT trained on any of the three domains, so each acts
as a fixed 'ruler' (cf. Inception in FID):
  - DINOv2 ViT-S/14  : strong self-supervised foundation features (384-d)
  - COCO-init YOLOv11s backbone : the paper's exact architecture (GAP of SPPF)

Both return an (N, d) float32 array of L2-normalized embeddings.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_batch(imgs, size):
    """list of HxWx3 uint8 -> (N,3,size,size) float in [0,1]."""
    out = []
    for a in imgs:
        im = Image.fromarray(a).resize((size, size), Image.BILINEAR)
        out.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


class DinoV2Encoder:
    dim = 384

    def __init__(self, device=None):
        self.device = device or pick_device()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, imgs, size=224, batch=32):
        feats = []
        for i in range(0, len(imgs), batch):
            x = _to_batch(imgs[i:i + batch], size)
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            f = self.model(x.to(self.device))          # (B, 384) CLS features
            feats.append(f.float().cpu())
        f = torch.cat(feats)
        return F.normalize(f, dim=1).numpy()


class YoloBackboneEncoder:
    def __init__(self, weights, device=None):
        from ultralytics import YOLO
        self.device = device or pick_device()
        self.model = YOLO(weights).model.eval().to(self.device)
        # hook the last SPPF (end of backbone) for a global descriptor
        sppf_idx = [i for i, m in enumerate(self.model.model)
                    if type(m).__name__ == "SPPF"][-1]
        self._feat = {}
        self.model.model[sppf_idx].register_forward_hook(
            lambda m, i, o: self._feat.__setitem__("f", o))
        self.dim = None

    @torch.no_grad()
    def embed(self, imgs, size=256, batch=16):
        feats = []
        for i in range(0, len(imgs), batch):
            x = _to_batch(imgs[i:i + batch], size).to(self.device)  # YOLO wants [0,1] RGB
            try:
                self.model(x)
            except Exception:
                pass  # we only need the hooked feature, not the detections
            fmap = self._feat["f"]                       # (B, C, h, w)
            v = fmap.mean(dim=(2, 3)).float().cpu()      # global average pool
            feats.append(v)
        f = torch.cat(feats)
        self.dim = f.shape[1]
        return F.normalize(f, dim=1).numpy()
