import argparse
import torch
from pathlib import Path
from common import *
import checks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Шлях до папки з seg_train / seg_test")
    ap.add_argument("--skip-checks", action="store_true", help="Пропустити перевірки (якщо вони вже пройдені)")
    ap.add_argument("--checks-only", action="store_true", help="Виконати лише перевірки пунктів 1–2 і зупинитись")
    a = ap.parse_args()

    device = get_device()
    print("Пристрій:", device)

    # 1. Поділ даних (split.csv)
    build_split(a.data)
    rows = read_split()

    # 2. Перевірки (пункти 1–2)
    if not a.skip_checks:
        print("\nЗапуск перевірок...")
        checks.run_all(a.data, rows, device)
    if a.checks_only:
        print("\n--checks-only: зупиняємось перед основними запусками.")
        raise SystemExit(0)

    # 3. Сім запусків (пункт 3)
    print("\nПочаток семи запусків...")
    for regime, seed in RUNS:
        run_one(regime, seed, a.data, rows, device)

    # 4. Вибір конфігурації за validation і збереження моделі (пункт 5)
    logs = load_logs()
    reg, seed, mean, F = select_best(logs)
    sd = torch.load(Path("runs") / f"{reg}_seed{seed}" / "model.pt", map_location="cpu")
    torch.save({"state_dict": sd, "regime": reg, "seed": seed,
                "classes": CLASSES, "img_size": IMG}, "lab3_model.pt")
    print(f"\nВибрано: режим {reg}, зерно {seed}")
    print(f"Середній F1 на validation за режимами: {mean}")
    print("Модель збережено у lab3_model.pt")