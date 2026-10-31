import csv, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN, STD, IMG, EPOCHS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 160, 8
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
REGIMES = ["scratch", "frozen", "unfreeze"]
# модулі, які під час навчання переводяться в eval() (стовпець «У eval()» таблиці режимів)
EVAL_MODULES = {"scratch": [],
                "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
                "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"]}
# модулі верхнього рівня, що навчаються (requires_grad=True); None означає «усе»
TRAINABLE = {"scratch": None, "frozen": ["fc"], "unfreeze": ["layer4", "fc"]}
SPLIT = Path("split.csv")


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- поділ ----------------
def _sorted_files(root, sub, c):
    # імена файлів сортуються як рядки (так само, як sorted() та ImageFolder)
    return sorted(p.name for p in (Path(root) / sub / sub / c).glob("*.jpg"))


def build_split(root, out=SPLIT):
    rows = []
    rng = np.random.default_rng(0)                  # один генератор для всіх класів seg_train
    for c in CLASSES:
        files = _sorted_files(root, "seg_train", c)
        idx = np.arange(len(files)); rng.shuffle(idx)
        rows += [(files[i], c, "train") for i in sorted(idx[:200])]
    rng = np.random.default_rng(0)                  # новий генератор для seg_test
    for c in CLASSES:
        files = _sorted_files(root, "seg_test", c)
        n = len(files); idx = np.arange(n); rng.shuffle(idx); cut = n // 2
        rows += [(files[i], c, "validation") for i in sorted(idx[:cut])]
        rows += [(files[i], c, "test") for i in sorted(idx[cut:])]
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "label", "part"]); w.writerows(rows)


def read_split(path=SPLIT):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def part_rows(rows, part):
    return [r for r in rows if r["part"] == part]


# ---------------- дані ----------------
def train_tf():
    return T.Compose([T.RandomResizedCrop(IMG, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_tf():
    return T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, rows, root, tf, cache=False):
        self.rows, self.root, self.tf = rows, Path(root), tf
        self.cache = {} if cache else None          # кеш лише для детермінованих перетворень (validation/test)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        if self.cache is not None and i in self.cache: return self.cache[i]
        r = self.rows[i]
        sub = "seg_train" if r["part"] == "train" else "seg_test"
        img = Image.open(self.root / sub / sub / r["label"] / r["file"]).convert("RGB")
        item = (self.tf(img), CLASSES.index(r["label"]))
        if self.cache is not None: self.cache[i] = item
        return item


def make_loader(rows, root, part):
    pr = part_rows(rows, part)
    if part == "train":
        # без власного generator: перемішування бере випадковість із глобального генератора PyTorch
        return DataLoader(DS(pr, root, train_tf()), batch_size=32, shuffle=True,
                          drop_last=False, num_workers=0)
    return DataLoader(DS(pr, root, eval_tf(), cache=True), batch_size=64,
                      shuffle=False, num_workers=0)


# ---------------- модель ----------------
def build_model(regime, seed):
    torch.manual_seed(seed); np.random.seed(seed)                   # крок 1
    W = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=W)                                      # крок 2
    model.fc = nn.Linear(512, 6)                                     # крок 3
    if TRAINABLE[regime] is not None:                                # крок 4
        for p in model.parameters(): p.requires_grad = False
        for name in TRAINABLE[regime]:
            for p in getattr(model, name).parameters(): p.requires_grad = True
    return model


def make_optimizer(model, regime):                                   # крок 5 (після .to(device))
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": 3e-3}]
    elif regime == "frozen":
        groups = [{"params": list(model.fc.parameters()), "lr": 3e-3}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": 3e-3},
                  {"params": list(model.layer4.parameters()), "lr": 3e-4}]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def set_train_mode(model, regime):
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = [model(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def labels_of(rows, part):
    return np.array([CLASSES.index(r["label"]) for r in part_rows(rows, part)])


def macro_f1(y, logits):
    return float(f1_score(y, logits.argmax(1), average="macro"))


def git_info():
    """Хеш коміту і прапорець «є незакомічені зміни КОДУ» (без untracked-файлів: split.csv, runs/)."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        d = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no",
             "--", "*.py", "pyproject.toml", "uv.lock"], text=True).strip())
        return h, d
    except Exception:
        return None, None


# ---------------- один запуск ----------------
def run_one(regime, seed, root, rows, device):
    import torchvision, sys
    name = f"{regime}_seed{seed}"
    out = Path("runs") / name; out.mkdir(parents=True, exist_ok=True)
    model = build_model(regime, seed).to(device)
    opt = make_optimizer(model, regime)
    crit = nn.CrossEntropyLoss()
    tr, va = make_loader(rows, root, "train"), make_loader(rows, root, "validation")
    assert len(tr) == 38, f"очікувалось 38 кроків за епоху, отримано {len(tr)}"
    yv = labels_of(rows, "validation")
    losses, f1s = [], []
    train_secs = 0.0                                  # лише цикл навчання, без валідації
    t_all = time.perf_counter()
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        set_train_mode(model, regime)
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)  # .item() синхронізує пристрій
        train_secs += time.perf_counter() - t0
        losses.append(tot / n)
        lg = predict(model, va, device)
        f1s.append(macro_f1(yv, lg))
        print(f"{name} епоха {ep+1}: втрата_train={losses[-1]:.4f} F1_валід.={f1s[-1]:.4f}", flush=True)
    total_secs = time.perf_counter() - t_all
    np.save(out / "val_logits.npy", lg)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    h, dirty = git_info()
    cfg = {
        "regime": regime, "seed": seed, "epochs": EPOCHS, "steps_per_epoch": len(tr),
        "steps_total": len(tr) * EPOCHS, "batch_size": 32, "eval_batch_size": 64,
        "dtype": "float32", "img_size": IMG,
        "transforms": {"train": "RandomResizedCrop(160,scale=(0.7,1.0))->RandomHorizontalFlip()->ToTensor->Normalize",
                       "eval": "Resize((160,160))->ToTensor->Normalize", "mean": MEAN, "std": STD},
        "model": "resnet18", "weights": None if regime == "scratch" else "IMAGENET1K_V1",
        "trainable": TRAINABLE[regime], "eval_modules": EVAL_MODULES[regime],
        "optimizer": {"name": "AdamW", "weight_decay": 1e-4,
                      "defaults": "betas=(0.9,0.999), eps=1e-8, amsgrad=False",
                      "groups": [{"lr": g["lr"], "n_params": sum(p.numel() for p in g["params"])}
                                 for g in opt.param_groups]},
        "loss": "CrossEntropyLoss(mean)", "bn": "momentum=0.1, eps=1e-5 (за замовчуванням)",
        "loader": {"train": "bs=32, shuffle=True, drop_last=False, num_workers=0",
                   "eval": "bs=64, shuffle=False, num_workers=0"},
        "no_lr_scheduler_clip_amp_smoothing_classweights_earlystop": True,
        "git_commit": h, "git_dirty": dirty, "split_sha256": sha256_file(SPLIT),
        "python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        "train_loss": losses, "val_f1": f1s,
        "train_seconds": train_secs, "total_seconds_with_validation": total_secs,
        "train_seconds_note": "train_seconds = лише цикл навчання (без валідації)",
    }
    (out / "run.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------- вибір (спільний для train і analyze) ----------------
def load_logs():
    return {(r, s): json.loads((Path("runs") / f"{r}_seed{s}" / "run.json").read_text(encoding="utf-8"))
            for r, s in RUNS}


def select_best(logs):
    F = {k: v["val_f1"][-1] for k, v in logs.items()}
    mean = {reg: float(np.mean([v for (r, s), v in F.items() if r == reg])) for reg in REGIMES}
    best_reg = max(mean, key=mean.get)
    best_seed = max((s for (r, s) in F if r == best_reg), key=lambda s: F[(best_reg, s)])
    return best_reg, best_seed, mean, F