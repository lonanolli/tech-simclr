import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images_under_train(root: str) -> List[str]:
    imgs = []
    for cls in sorted(os.listdir(root)):
        cls_images_dir = os.path.join(root, cls, "images")
        if not os.path.isdir(cls_images_dir):
            continue
        for fn in sorted(os.listdir(cls_images_dir)):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                imgs.append(os.path.join(cls_images_dir, fn))
    return imgs


def list_images_under_val(root: str) -> List[str]:
    img_dir = os.path.join(root, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Expected val/images/ under: {root}")
    imgs = [
        os.path.join(img_dir, fn)
        for fn in sorted(os.listdir(img_dir))
        if fn.lower().endswith((".jpeg", ".jpg", ".png"))
    ]
    return imgs


# ----------------------------
# Data
# ----------------------------
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, train_root: str, transform: Optional[object] = None):
        self.train_root = train_root
        self.transform = transform
        self.imgs = list_images_under_train(train_root)
        if len(self.imgs) == 0:
            raise RuntimeError(f"No images found under: {train_root}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is None:
            x1 = T.ToTensor()(img)
            x2 = T.ToTensor()(img)
        else:
            x1 = self.transform(img)
            x2 = self.transform(img)
        return x1, x2


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, val_root: str, transform: Optional[object] = None):
        self.val_root = val_root
        self.transform = transform
        self.imgs = list_images_under_val(val_root)
        if len(self.imgs) == 0:
            raise RuntimeError(f"No images found under: {val_root}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        original = T.ToTensor()(img)
        x = self.transform(img) if self.transform else original
        return original, x


def get_train_transforms(img_size: int = 96):
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def get_test_transforms():
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SimCLRModel(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim, bias=True),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:

    assert z1.shape == z2.shape
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)

    sim = (z @ z.T) / temperature
    sim = sim - torch.eye(2 * n, device=z.device) * 1e9

    pos_idx = torch.arange(n, device=z.device)
    positives = torch.cat([sim[pos_idx, pos_idx + n], sim[pos_idx + n, pos_idx]], dim=0)

    denom = torch.logsumexp(sim, dim=1)
    loss = -positives + denom
    return loss.mean()


@torch.no_grad()
def extract_embeddings(model, loader, device, max_batches: int = 1):
    model.eval()
    all_emb = []
    all_imgs = []
    batches = 0
    for orig, x in loader:
        x = x.to(device)
        _, z = model(x)
        all_emb.append(z.cpu())
        all_imgs.append(orig)
        batches += 1
        if batches >= max_batches:
            break
    return torch.cat(all_emb, dim=0), torch.cat(all_imgs, dim=0)


def train(model, train_loader, device, epochs: int, lr: float, temperature: float, target_loss: float):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for ep in range(epochs):
        total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for x1, x2 in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total / len(train_loader)
        losses.append(avg)
        print(f"Epoch {ep+1}: avg loss = {avg:.4f}")

        if avg < target_loss:
            print(f"Target loss {target_loss} achieved, stopping early.")
            break

    return losses


def plot_embeddings_thumbnails_tsne(model, test_loader, device, perplexity: int = 30):

    emb, imgs = extract_embeddings(model, test_loader, device, max_batches=1)
    emb_2d = TSNE(n_components=2, perplexity=perplexity).fit_transform(emb.numpy())

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Embeddings t-SNE (thumbnails)")
    ax.set_xticks([])
    ax.set_yticks([])

    # scale points for easier thumbnail placement
    x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
    y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
    sx = (emb_2d[:, 0] - x_min) / (x_max - x_min + 1e-9)
    sy = (emb_2d[:, 1] - y_min) / (y_max - y_min + 1e-9)

    for i in range(emb_2d.shape[0]):
        img = imgs[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        im = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(im, (sx[i], sy[i]), frameon=False)
        ax.add_artist(ab)

    plt.tight_layout()
    plt.show()


def plot_class_separation_tsne(model, train_root: str, device, k: int = 2, samples_per_class: int = 20):

    class_dirs = [d for d in sorted(os.listdir(train_root)) if os.path.isdir(os.path.join(train_root, d))]
    class_dirs = class_dirs[:k]

    test_tf = get_test_transforms()

    embs = []
    labels = []

    model.eval()
    with torch.no_grad():
        for ci, cls in enumerate(class_dirs):
            img_dir = os.path.join(train_root, cls, "images")
            imgs = [fn for fn in sorted(os.listdir(img_dir)) if fn.lower().endswith((".jpeg", ".jpg", ".png"))]
            imgs = imgs[:samples_per_class]
            for fn in imgs:
                img = Image.open(os.path.join(img_dir, fn)).convert("RGB")
                x = test_tf(img).unsqueeze(0).to(device)
                _, z = model(x)
                embs.append(z.squeeze(0).cpu().numpy())
                labels.append(ci)

    embs = np.stack(embs, axis=0)
    labels = np.array(labels)

    emb_2d = TSNE(n_components=2, random_state=42).fit_transform(embs)
    plt.figure(figsize=(8, 6))
    for ci, cls in enumerate(class_dirs):
        mask = labels == ci
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=cls, alpha=0.7)
    plt.legend()
    plt.title("Class separation (t-SNE)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.show()


def plot_nearest_neighbors(model, test_loader, device, num_queries: int = 3, k: int = 6):

    emb, imgs = extract_embeddings(model, test_loader, device, max_batches=1)
    emb = emb.to(device)

    B = emb.shape[0]
    query_idx = random.sample(range(B), num_queries)
    query = emb[query_idx]

    sim = query @ emb.T
    topk_idx = torch.topk(sim, k=k, dim=1).indices.cpu().numpy()

    def imshow(img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()
        plt.imshow(np.clip(img, 0, 1))
        plt.axis("off")

    for qi, idx0 in enumerate(query_idx):
        plt.figure(figsize=(14, 3))
        for j, idx in enumerate(topk_idx[qi]):
            plt.subplot(1, k, j + 1)
            imshow(imgs[idx])
            plt.title("Query" if j == 0 else f"Top {j}")
        plt.tight_layout()
        plt.show()


def plot_losses(losses: List[float]):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Training loss (NT-Xent)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Path to tiny-imagenet-200")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--target-loss", type=float, default=3.0)
    p.add_argument("--num-workers", type=int, default=2)

    # viz params
    p.add_argument("--tsne-perplexity", type=int, default=30)
    p.add_argument("--class-k", type=int, default=2)
    p.add_argument("--samples-per-class", type=int, default=20)
    p.add_argument("--nn-queries", type=int, default=3)
    p.add_argument("--nn-topk", type=int, default=6)

    # save/load
    p.add_argument("--save", type=str, default="simclr.pt")
    p.add_argument("--load", type=str, default="")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")

    train_tf = get_train_transforms(img_size=96)
    test_tf = get_test_transforms()

    train_ds = TrainDataset(train_root, transform=train_tf)
    val_ds = TestDataset(val_root, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SimCLRModel(embedding_dim=args.embedding_dim).to(device)

    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.load}")

    losses = train(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        target_loss=args.target_loss,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "losses": losses,
        },
        args.save,
    )
    print(f"Saved checkpoint to: {args.save}")

    plot_losses(losses)
    plot_embeddings_thumbnails_tsne(model, test_loader, device, perplexity=args.tsne_perplexity)
    plot_class_separation_tsne(
        model, train_root=train_root, device=device, k=args.class_k, samples_per_class=args.samples_per_class
    )
    plot_nearest_neighbors(
        model, test_loader=test_loader, device=device, num_queries=args.nn_queries, k=args.nn_topk
    )


if __name__ == "__main__":
    main()
