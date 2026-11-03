"""Extrai embeddings congelados com outros backbones para o scratchpad."""
import sys, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import fs_lab as L
from proto_sgc.data import load_local_meta_album, ImageRecordDataset

def backbone(name):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Identity()
    elif name == "dinov2s":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif name == "convnext_t":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT); m.classifier[2] = nn.Identity()
    return m.eval().requires_grad_(False)

def main(bb, size, names):
    m = backbone(bb).to(L.DEVICE)
    resize = [transforms.Resize((size, size), antialias=True)]
    tf = transforms.Compose(resize + [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    for name in names:
        out = L.LAB/f"feats/{name}_{bb}_{size}.pt"
        out.parent.mkdir(exist_ok=True)
        if out.exists(): continue
        recs = load_local_meta_album(L.data_root(name))
        classes = sorted({r.label for r in recs}); cid = {c: i for i, c in enumerate(classes)}
        dl = DataLoader(ImageRecordDataset(recs, tf, cid), batch_size=32, num_workers=2)
        fs, ys = [], []
        with torch.inference_mode():
            for x, y in dl: fs.append(m(x.to(L.DEVICE)).float().cpu()); ys.append(y)
        f, y = torch.cat(fs), torch.cat(ys)
        torch.save({c: f[y == cid[c]].contiguous() for c in classes}, out)
        print(name, bb, size, tuple(f.shape), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:] or L.NAMES)
