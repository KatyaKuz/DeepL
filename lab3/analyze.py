import argparse, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18
from common import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Шлях до папки з seg_train / seg_test")
    a = ap.parse_args()

    # --- 1. Перевірка хешу split.csv ---
    logs = load_logs()
    cur = sha256_file(SPLIT)
    hashes = {v["split_sha256"] for v in logs.values()}
    if hashes != {cur}:
        raise SystemExit(f"Хеш split.csv не збігається з журналами: {hashes} проти {cur}")
    rows = read_split()

    # --- 2. Таблиця семи запусків ---
    print("\n# | Режим | Зерно | Втрата train (епоха 8) | macro-F1 validation (епоха 8) | Час навчання, с")
    for i, (r, s) in enumerate(RUNS, 1):
        v = logs[(r, s)]
        print(f"{i} | {r} | {s} | {v['train_loss'][-1]:.4f} | {v['val_f1'][-1]:.4f} | {v['train_seconds']:.0f}")

    # --- 3. Криві ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, EPOCHS + 1)
    col = {"scratch": "tab:red", "frozen": "tab:blue", "unfreeze": "tab:green"}
    for (r, s), v in logs.items():
        ls = ["-", "--", ":"][s - 1]
        ax[0].plot(ep, v["train_loss"], color=col[r], alpha=.8, label=f"{r} зерно {s}", ls=ls)
        ax[1].plot(ep, v["val_f1"], color=col[r], alpha=.8, label=f"{r} зерно {s}", ls=ls)
    ax[0].set(title="Втрата на train", xlabel="епоха", ylabel="втрата")
    ax[1].set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1")
    ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig("fig_curves.png", dpi=130); plt.close()

    # --- 4. Розклад ефектів ---
    F = {reg: np.array([logs[(r, s)]["val_f1"][-1] for r, s in RUNS if r == reg]) for reg in REGIMES}
    Fs = F["scratch"][0]
    Ff, Fu = F["frozen"].mean(), F["unfreeze"].mean()
    Rf, Ru = np.ptp(F["frozen"]), np.ptp(F["unfreeze"])
    R = max(Rf, Ru)
    dpre, dunf = Ff - Fs, Fu - Ff
    print(f"\nF_scratch,1={Fs:.4f}  F_frozen(сер.)={Ff:.4f}  F_unfreeze(сер.)={Fu:.4f}")
    print(f"R_frozen={Rf:.4f}  R_unfreeze={Ru:.4f}  R={R:.4f}")
    print(f"Δpre={dpre:+.4f} ({'|Δ|>R: ефект вартий перевірки, але не доведений' if abs(dpre) > R else '|Δ|<=R: не відрізняється від розкиду'})")
    print(f"Δunf={dunf:+.4f} ({'|Δ|>R: ефект вартий перевірки, але не доведений' if abs(dunf) > R else '|Δ|<=R: не відрізняється від розкиду'})")
    print(f"Δpre+Δunf={dpre + dunf:+.4f} = F_unfreeze(сер.) - F_scratch,1 = {Fu - Fs:+.4f}")

    # --- 5. Перевірка вибору та відповідності lab3_model.pt ---
    reg, seed, mean, _ = select_best(logs)
    ck = torch.load("lab3_model.pt", map_location="cpu")
    if ck["regime"] != reg or ck["seed"] != seed:
        raise SystemExit(f"lab3_model.pt ({ck['regime']}, {ck['seed']}) не відповідає правилу вибору ({reg}, {seed})")
    print(f"\nВибір: режим {reg}, зерно {seed}; середній F1 на validation: {mean}")
    if abs(Fu - Ff) <= R:
        print("Увага: |F_unfreeze(сер.) - F_frozen(сер.)| <= R, вибір між frozen і unfreeze визначив шум.")

    model = resnet18(weights=None); model.fc = nn.Linear(512, 6)
    model.load_state_dict(ck["state_dict"]); model.cpu().eval()

    # 5а. Точне порівняння ваг (лише якщо локальні ваги запусків існують; у репозиторії їх немає)
    sel = Path("runs") / f"{reg}_seed{seed}" / "model.pt"
    if sel.exists():
        st = torch.load(sel, map_location="cpu")
        if not all(torch.equal(ck["state_dict"][k], st[k]) for k in st):
            raise SystemExit("Ваги lab3_model.pt не збігаються з runs/.../model.pt")
        print("Ваги lab3_model.pt точно збігаються з вибраним запуском.")

    # 5б. Перерахунок validation на CPU та порівняння з val_logits.npy (працює і без model.pt)
    yv = labels_of(rows, "validation")
    vl = predict(model, make_loader(rows, a.data, "validation"), torch.device("cpu"))
    ref = np.load(Path("runs") / f"{reg}_seed{seed}" / "val_logits.npy")
    f_cpu, f_log = macro_f1(yv, vl), macro_f1(yv, ref)
    print(f"Validation: макс.|Δ логітів|={np.abs(vl - ref).max():.2e}; "
          f"macro-F1 на CPU={f_cpu:.4f} проти журналу={f_log:.4f}")
    if abs(f_cpu - f_log) > 5e-3:
        raise SystemExit("lab3_model.pt не відтворює validation-логіти вибраного запуску")

    # --- 6. Тест (один раз, на CPU) ---
    te = make_loader(rows, a.data, "test")
    logits = predict(model, te, torch.device("cpu"))
    yt = labels_of(rows, "test"); pred = logits.argmax(1)
    f1t = macro_f1(yt, logits); acc = float((pred == yt).mean())
    se = math.sqrt(acc * (1 - acc) / len(yt))
    print(f"\nTEST: macro-F1={f1t:.4f}, accuracy={acc:.4f}, SE≈{se:.4f}; "
          f"macro-F1 на validation вибраного запуску={logs[(reg, seed)]['val_f1'][-1]:.4f}")

    cm = confusion_matrix(yt, pred, labels=range(6))
    print("\nМатриця плутанини (рядки — справжній клас, стовпці — передбачений):")
    print("            " + " ".join(f"{c[:6]:>7s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:11s} " + " ".join(f"{cm[i, j]:7d}" for j in range(6)))

    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), CLASSES, rotation=45); ax.set_yticks(range(6), CLASSES)
    for i in range(6):
        for j in range(6): ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set(xlabel="передбачений клас", ylabel="справжній клас", title="Матриця плутанини (test)")
    plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=130); plt.close()

    # --- 7. П'ять характерних помилок ---
    test_rows = part_rows(rows, "test")
    prob = torch.softmax(torch.from_numpy(logits), 1).numpy()
    pairs = [(cm[i, j], i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))   # за спаданням кількості, потім за індексами класів
    files = {(i, j): sorted((k for k in range(len(test_rows)) if yt[k] == i and pred[k] == j),
                            key=lambda k: test_rows[k]["file"])   # сортування імен як рядків
             for _, i, j in pairs}
    use = pairs[:5]
    chosen, k = [], 0
    while len(chosen) < 5 and k < 50:
        for cnt, i, j in use:
            if len(chosen) < 5 and len(files[(i, j)]) > k:
                chosen.append((files[(i, j)][k], cnt, i, j))
        k += 1

    print("\nФайл | Справжній клас | Передбачений клас | Імовірність передбаченого | Кількість таких помилок")
    fig, ax = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4.3))
    ax = np.atleast_1d(ax)
    for a_, (idx, cnt, i, j) in zip(ax, chosen):
        r = test_rows[idx]
        print(f"{r['file']} | {CLASSES[i]} | {CLASSES[j]} | {prob[idx, j]:.3f} | {cnt}")
        img = Image.open(Path(a.data) / "seg_test" / "seg_test" / r["label"] / r["file"]).convert("RGB")
        a_.imshow(img); a_.axis("off")
        a_.set_title(f"{r['file']}\nсправжній: {CLASSES[i]}\nпередбачений: {CLASSES[j]} ({prob[idx, j]:.2f})", fontsize=9)
    plt.tight_layout(); plt.savefig("fig_errors.png", dpi=130)