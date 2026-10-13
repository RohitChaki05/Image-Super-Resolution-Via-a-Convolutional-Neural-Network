"""SRCNN 2x image super-resolution (reimplementation of Garber et al., CS229 2020).

Usage:
  python srcnn.py train --data_dir DIV2K_valid_HR --out_dir results --epochs 100
  python srcnn.py demo  --image some.jpg --ckpt results/best_srcnn.pt --out_dir results
"""
import argparse, glob, json, os, random
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}
SIZE_CROP, SIZE_HR, SIZE_LR = 800, 224, 112


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SRCNN(nn.Module):
    """9x9 (3->64) + ReLU, 1x1 (64->32) + ReLU, 5x5 (32->3). Input is the bicubic-upscaled image."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def make_pair(img):
    """Center crop (up to 800x800) -> 224x224 target -> 112x112 -> back to 224x224."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(SIZE_CROP, w, h)
    l, t = (w - s) // 2, (h - s) // 2
    hr = img.crop((l, t, l + s, t + s)).resize((SIZE_HR, SIZE_HR), Image.BICUBIC)
    lr = hr.resize((SIZE_LR, SIZE_LR), Image.BICUBIC)
    return hr, lr


def psnr(a, b):
    """Per-image PSNR in dB on RGB tensors in [0,1] (MAX = 1)."""
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def build_dataset(data_dir, n_images=100):
    files = sorted(glob.glob(os.path.join(data_dir, "*.png")) + glob.glob(os.path.join(data_dir, "*.jpg")))[:n_images]
    if len(files) == 0:
        raise SystemExit(f"No images found in {data_dir}")
    hr, ups = [], {m: [] for m in MODES}
    for f in files:
        h, l = make_pair(Image.open(f))
        hr.append(to_tensor(h))
        for m, mode in MODES.items():
            ups[m].append(to_tensor(l.resize((SIZE_HR, SIZE_HR), mode)))
    return torch.stack(hr), {m: torch.stack(v) for m, v in ups.items()}, files


def evaluate(model, x, y, device, bs=16):
    model.eval(); ps, ms = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out = model(x[i:i + bs].to(device)).clamp(0, 1).cpu()
            yy = y[i:i + bs]
            ps.append(psnr(out, yy)); ms.append(((out - yy) ** 2).mean(dim=(1, 2, 3)))
    return torch.cat(ps).mean().item(), torch.cat(ms).mean().item()


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(args.out_dir, exist_ok=True)
    hr, ups, files = build_dataset(args.data_dir, args.n_images)
    n = len(hr)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    x, y = ups["bicubic"], hr
    model = SRCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(x[tr], y[tr])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    hist = {"train_loss": [], "val_loss": [], "val_psnr": []}
    best, ckpt = -1.0, os.path.join(args.out_dir, "best_srcnn.pt")
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        vp, vm = evaluate(model, x[va], y[va], device)
        hist["train_loss"].append(tot / len(ds)); hist["val_loss"].append(vm); hist["val_psnr"].append(vp)
        if vp > best:
            best = vp; torch.save(model.state_dict(), ckpt)
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d} train_mse {hist['train_loss'][-1]:.5f} val_mse {vm:.5f} val_psnr {vp:.2f} dB")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    results = {"seed": args.seed, "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
               "best_val_psnr": best}
    results["test_psnr_srcnn"], results["test_mse_srcnn"] = evaluate(model, x[te], y[te], device)
    for m in MODES:
        results[f"test_psnr_{m}"] = psnr(ups[m][te], y[te]).mean().item()
        results[f"test_mse_{m}"] = ((ups[m][te] - y[te]) ** 2).mean().item()
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE"); ax[0].legend()
    ax[1].plot(hist["val_psnr"], label="SRCNN val")
    ax[1].axhline(psnr(ups["bicubic"][va], y[va]).mean().item(), color="r", ls="--", label="bicubic val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("PSNR (dB)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "curves.png"), dpi=150); plt.close()

    # Sample test images: bicubic input | SRCNN | target
    k = min(3, len(te))
    fig, ax = plt.subplots(k, 3, figsize=(9, 3 * k))
    ax = np.atleast_2d(ax)
    model.eval()
    with torch.no_grad():
        for r in range(k):
            i = te[r].item()
            out = model(x[i:i + 1].to(device)).clamp(0, 1).cpu()
            for c, (im, t) in enumerate([(x[i:i + 1], "Bicubic"), (out, "SRCNN"), (y[i:i + 1], "Target")]):
                ax[r, c].imshow(im[0].permute(1, 2, 0).numpy()); ax[r, c].axis("off")
                ax[r, c].set_title(t if t == "Target" else f"{t} ({psnr(im, y[i:i+1]).item():.2f} dB)", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "samples.png"), dpi=150); plt.close()
    print("Saved results.json, curves.png, samples.png, best_srcnn.pt to", args.out_dir)


def demo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()
    hr, lr = make_pair(Image.open(args.image))
    bic = to_tensor(lr.resize((SIZE_HR, SIZE_HR), Image.BICUBIC)).unsqueeze(0)
    tgt = to_tensor(hr).unsqueeze(0)
    with torch.no_grad():
        out = model(bic.to(device)).clamp(0, 1).cpu()
    pb, ps = psnr(bic, tgt).item(), psnr(out, tgt).item()
    print(f"Bicubic PSNR: {pb:.2f} dB | SRCNN PSNR: {ps:.2f} dB")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    for a, im, t in zip(ax, [bic, out, tgt], [f"Bicubic ({pb:.2f} dB)", f"SRCNN ({ps:.2f} dB)", "Target"]):
        a.imshow(im[0].permute(1, 2, 0).numpy()); a.set_title(t); a.axis("off")
    path = os.path.join(args.out_dir, "demo_output.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print("Saved", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data_dir", required=True); t.add_argument("--out_dir", default="results")
    t.add_argument("--epochs", type=int, default=100); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch_size", type=int, default=16); t.add_argument("--n_images", type=int, default=100)
    t.add_argument("--seed", type=int, default=42)
    d = sub.add_parser("demo")
    d.add_argument("--image", required=True); d.add_argument("--ckpt", default="results/best_srcnn.pt")
    d.add_argument("--out_dir", default="results")
    a = p.parse_args()
    train(a) if a.cmd == "train" else demo(a)
