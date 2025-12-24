# train_itransformer_label10.py
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import list_snapshot_files, split_files_by_date, SnapshotWindowDataset
from model_itransformer import iTransformerClassifier
from metrics import accuracy, fbeta_macro
from dataset_cached import CachedWindowDataset

def eval_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in tqdm(loader, desc="eval", leave=False):
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = ce(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            ys.append(y.cpu().numpy())
            ps.append(pred)

    y_true = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps, axis=0) if ps else np.array([], dtype=np.int64)
    return {
        "loss": loss_sum / max(1, n),
        "acc": accuracy(y_true, y_pred) if n > 0 else 0.0,
        "f05": fbeta_macro(y_true, y_pred, beta=0.5, num_classes=3) if n > 0 else 0.0,
    }

def train(
    data_dir: str,
    out_dir: str,
    test_start: int = 70,
    test_end: int = 78,
    val_days: int = 10,
    window: int = 100,
    batch_size: int = 512,
    num_workers: int = 8,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
    cache_dir: str = "",
):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    files = list_snapshot_files(data_dir)
    train_files, val_files, test_files = split_files_by_date(files, test_start, test_end, val_days=val_days)
    print(f"[Files] total={len(files)} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # train_ds = SnapshotWindowDataset(train_files, label_col="label_10", window=window, cache_files=8)
    # val_ds   = SnapshotWindowDataset(val_files,   label_col="label_10", window=window, cache_files=8)
    # test_ds  = SnapshotWindowDataset(test_files,  label_col="label_10", window=window, cache_files=8)
    train_ds = CachedWindowDataset(cache_dir, train_files, window=window)
    val_ds   = CachedWindowDataset(cache_dir, val_files,   window=window)
    test_ds  = CachedWindowDataset(cache_dir, test_files,  window=window)

    print(f"[Samples] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    n_vars = train_ds.n_features
    print(f"[Features] F={n_vars}  window={window}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = iTransformerClassifier(n_vars=n_vars, seq_len=window, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    best_f05 = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    meta_path = os.path.join(out_dir, "meta.json")
    log_path = os.path.join(out_dir, "train_log.jsonl")

    # 保存meta（推理/复现实验用）
    meta = {
        "label": "label_10",
        "window": window,
        "feature_cols": train_ds.feature_cols,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "preprocess": "exp_log (size: expm1(100*x), amount_delta: log1p, price: identity)",
        "split": {"test_start": test_start, "test_end": test_end, "val_days": val_days},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)

        train_loss = loss_sum / max(1, n)
        val_metrics = eval_epoch(model, val_loader, device)
        test_metrics = eval_epoch(model, test_loader, device)

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "time_sec": dt,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f05": val_metrics["f05"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_f05": test_metrics["f05"],
        }
        print(json.dumps(row, ensure_ascii=False))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # early stopping by val_f05
        if val_metrics["f05"] > best_f05 + 1e-6:
            best_f05 = val_metrics["f05"]
            bad_epochs = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs. best_val_f05={best_f05:.6f}")
                break

    print(f"[Done] best model saved to: {best_path}")
    print(f"[Done] meta saved to: {meta_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="out_label10_itransformer")
    ap.add_argument("--test_start", type=int, default=70)
    ap.add_argument("--test_end", type=int, default=78)
    ap.add_argument("--val_days", type=int, default=10)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cache_dir", required=True)
    args = ap.parse_args()

    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        test_start=args.test_start,
        test_end=args.test_end,
        val_days=args.val_days,
        window=args.window,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        cache_dir=args.cache_dir,
    )
