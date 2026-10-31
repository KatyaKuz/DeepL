import itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, torch.nn as nn
from common import *

EXPECTED = {  # клас: [train, validation, test] = (кількість, сума номерів)
    "buildings": [(200, 1996761), (218, 4837323), (219, 4837559)],
    "forest":    [(200, 2032625), (237, 5276962), (237, 5291326)],
    "glacier":   [(200, 2183923), (276, 6135652), (277, 6157250)],
    "mountain":  [(200, 1950345), (262, 5789535), (263, 5849765)],
    "sea":       [(200, 1987613), (255, 5655068), (255, 5668011)],
    "street":    [(200, 1841817), (250, 5541845), (251, 5547533)],
}
PARTS = ["train", "validation", "test"]

# (кількість параметрів у групі, швидкість навчання); для unfreeze порядок: fc, layer4
EXPECTED_GROUPS = {
    "scratch":  [(11179590, 3e-3)],
    "frozen":   [(3078, 3e-3)],
    "unfreeze": [(3078, 3e-3), (8393728, 3e-4)],
}
EXPECTED_TRAINABLE = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}


def check_split(rows):
    ok = True
    print("\n== Контроль поділу (кількість / сума номерів файлів) ==")
    for c in CLASSES:
        cells = []
        for part, exp in zip(PARTS, EXPECTED[c]):
            fs = [r["file"] for r in rows if r["label"] == c and r["part"] == part]
            got = (len(fs), sum(int(f.split(".")[0]) for f in fs))
            ok &= (got == exp)
            cells.append(f"{got[0]}/{got[1]} {'збігається' if got == exp else 'НЕ ЗБІГАЄТЬСЯ (очікувано %s)' % (exp,)}")
        print(f"{c:10s} | " + " | ".join(cells))
    sets = {p: {r["file"] for r in rows if r["part"] == p} for p in PARTS}
    disjoint = all(not (sets[a] & sets[b]) for a, b in itertools.combinations(PARTS, 2))
    print("Файли не перетинаються між частинами:", "так" if disjoint else "НІ")
    return ok and disjoint


def check_batch(rows, root, device):
    build_model("scratch", 1)
    x, y = next(iter(make_loader(rows, root, "train")))
    mean = torch.tensor(MEAN).view(3, 1, 1); std = torch.tensor(STD).view(3, 1, 1)
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for i, a in enumerate(ax.flat):
        a.imshow((x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        a.set_title(CLASSES[y[i]], fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig("fig_batch.png", dpi=120); plt.close()
    print("\nЗбережено fig_batch.png")


def check_init_loss(rows, root, device):
    print(f"\n== Початкова втрата (ln 6 = {np.log(6):.4f}) ==")
    for regime in ["scratch", "frozen"]:
        model = build_model(regime, 1).to(device)
        x, y = next(iter(make_loader(rows, root, "train")))
        set_train_mode(model, regime)
        with torch.no_grad():
            loss = nn.CrossEntropyLoss()(model(x.to(device)), y.to(device)).item()
        print(f"{regime}: L0 = {loss:.4f}")


def check_params(device):
    print("\n== Параметри ==")
    ok = True
    for regime in REGIMES:
        model = build_model(regime, 1).to(device)
        opt = make_optimizer(model, regime)
        tot = sum(p.numel() for p in model.parameters())
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        leak = any(not p.requires_grad and id(p) in in_opt for p in model.parameters())
        groups = [(sum(p.numel() for p in g["params"]), g["lr"]) for g in opt.param_groups]
        wd_ok = all(g["weight_decay"] == 1e-4 for g in opt.param_groups)
        good = (tot == 11179590 and tr == EXPECTED_TRAINABLE[regime] and not leak
                and groups == EXPECTED_GROUPS[regime] and wd_ok)
        print(f"{regime}: усього={tot}, навчаються={tr}, групи (к-сть, lr)={groups}, "
              f"weight_decay коректний={wd_ok}, заморожені параметри в оптимізаторі={leak} "
              f"-> {'ОК' if good else 'ПОМИЛКА'}")
        ok &= good
    return ok


STAT = ("running_mean", "running_var", "num_batches_tracked")


def bn_experiment(regime, rows, root, device, wrong=False):
    model = build_model(regime, 1).to(device)
    opt = make_optimizer(model, regime)
    loader = make_loader(rows, root, "train")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if wrong: model.train()
    else: set_train_mode(model, regime)
    crit = nn.CrossEntropyLoss()
    for x, y in itertools.islice(loader, 5):
        opt.zero_grad(); crit(model(x.to(device)), y.to(device)).backward(); opt.step()
    after = model.state_dict()
    ok = True
    print(f"\n== Що змінюється: {regime}{' (model.train() — навмисна помилка)' if wrong else ''} ==")
    print(f"{'модуль':8s} {'параметри':10s} {'BN-статистики':14s} збігається з очікуваним")
    for name, _ in model.named_children():
        keys = [k for k in before if k.split(".")[0] == name]
        if not keys: continue
        pk = [k for k in keys if k.split(".")[-1] not in STAT]
        sk = [k for k in keys if k.split(".")[-1] in STAT]
        pch = any(not torch.equal(before[k], after[k]) for k in pk)
        sch = any(not torch.equal(before[k], after[k]) for k in sk) if sk else None
        frozen = name in EVAL_MODULES[regime]
        match = (pch == (not frozen)) and (sch is None or sch == (not frozen))
        ok &= match
        print(f"{name:8s} {'так' if pch else 'ні':10s} "
              f"{('немає BN' if sch is None else 'так' if sch else 'ні'):14s} {'так' if match else 'ні'}")
    rm = (after["layer1.0.bn1.running_mean"] - before["layer1.0.bn1.running_mean"]).abs().max().item()
    rv = (after["layer1.0.bn1.running_var"] - before["layer1.0.bn1.running_var"]).abs().max().item()
    print(f"layer1.0.bn1: макс.|Δ running_mean|={rm:.3e}, макс.|Δ running_var|={rv:.3e}")
    return ok


def run_all(root, rows, device):
    ok = check_split(rows)
    check_batch(rows, root, device)
    check_init_loss(rows, root, device)
    ok &= check_params(device)
    ok &= bn_experiment("frozen", rows, root, device)
    ok &= bn_experiment("unfreeze", rows, root, device)
    print("\n>>> ДОСЛІД З НАВМИСНОЮ ПОМИЛКОЮ (model.train() замість set_train_mode). "
          "Прогноз має бути записаний у README ДО цього запуску.")
    bn_experiment("frozen", rows, root, device, wrong=True)   # очікується «ні» там, де в нормі «так»
    if not ok:
        raise SystemExit("Перевірки не збіглися з очікуваним — виправте помилку до основного експерименту.")
    print("\nУсі перевірки пройдено.")