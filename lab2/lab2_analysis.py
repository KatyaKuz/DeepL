"""ЛР2. Аналіз збережених прогнозів без повторного навчання (пункти 1–4).

Запуск:
    uv run python lab2_analysis.py --predictions results/predictions.npz --out analysis

Порядок роботи (важливий за умовою):
  1) незалежна перевірка пошуку порога на наборах A/B/C;
  2) validation: криві PR/ROC, ROC-AUC, повна таблиця кандидатів, t*,
     порівняння правил 0.5 / t* / +inf;
  3) фіксація t* у threshold.json (фізична ізоляція від test);
  4) лише після цього test: t* читається з файлу, test не впливає на поріг.

Округлення порогів — ЛИШЕ для відображення у консолі (через fmt_threshold).
У файлах (CSV, JSON) пороги зберігаються як float без округлення.

Артефакти в --out:
  split_summary.csv,
  verify_A.csv, verify_B.csv, verify_C.csv,
  pr_curve.png, roc_curve.png,
  validation_candidates.csv, validation_comparison.csv,
  threshold.json,
  test_summary.csv, test_comparison.csv, test_errors.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

INF = float("inf")
COST_FN = 10
COST_FP = 1
TABLE_COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]
COMPARISON_COLUMNS = ["rule", *TABLE_COLUMNS]


# ============================================================== дані
def load_predictions(path: Path) -> dict[str, np.ndarray]:
    """Завантажити збережені прогнози та мітки (без навчання й без CSV)."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def split_summary(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Кількість об'єктів, шахрайських транзакцій і їхня частка для кожної вибірки."""
    rows = []
    for name in ("train", "validation", "test"):
        y = data[f"{name}_y"]
        positives = int(np.count_nonzero(y == 1))
        rows.append(
            {
                "split": name,
                "n": len(y),
                "positives": positives,
                "positive_fraction": positives / len(y),
            }
        )
    return pd.DataFrame(rows)


# ============================================ метрики (власна реалізація)
def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    """TP, FP, FN, TN за правилом s >= t (пряме застосування правила)."""
    predicted = scores >= threshold
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    """C = 10*FN + FP (Amount у вартість не входить)."""
    return COST_FN * fn + COST_FP * fp


def safe_div(numerator: float, denominator: float) -> float | None:
    """Частка або None, якщо знаменник нульовий (метрика не визначена)."""
    return numerator / denominator if denominator > 0 else None


def print_metric(name: str, value: float | None, explanation: str) -> None:
    """Вивести метрику з поясненням, якщо значення не визначене."""
    if value is None:
        print(f"  {name}: не визначено ({explanation})")
    else:
        print(f"  {name}: {value:.6f}")


def fmt_threshold(t: float) -> str:
    """ТІЛЬКИ для відображення у консолі. У файлах пороги — без округлення."""
    return "+inf" if np.isinf(t) else f"{t:.8f}"


# ================ швидкий пошук: сортування + cumsum, O(n log n)
def table_fast(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Повна таблиця кандидатів: +inf та всі унікальні оцінки (за спаданням порога)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y_true[order]

    cum_tp = np.cumsum(sorted_y == 1)
    cum_fp = np.cumsum(sorted_y == 0)

    group_end = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])

    thresholds = np.r_[INF, sorted_scores[group_end]]
    tp = np.r_[0, cum_tp[group_end]]
    fp = np.r_[0, cum_fp[group_end]]

    total_pos = int(np.sum(y_true == 1))
    total_neg = len(y_true) - total_pos
    fn = total_pos - tp
    tn = total_neg - fp

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": COST_FN * fn + COST_FP * fp,
        }
    )


def pick_threshold(table: pd.DataFrame) -> float:
    """t*: мінімум C; при рівності — найбільший поріг"""
    thresholds = table["threshold"].to_numpy()
    assert np.all(thresholds[:-1] > thresholds[1:]), "пороги мають строго спадати"
    return float(thresholds[int(np.argmin(table["C"].to_numpy()))])


# ---------- повільний перебір: ЛИШЕ для перевірки на малих наборах A/B/C
def table_brute(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Незалежна таблиця: для кожного кандидата пряме застосування s >= t."""
    candidates = sorted({float(s) for s in scores} | {INF}, reverse=True)
    rows = []
    for t in candidates:
        tp, fp, fn, tn = confusion(y_true, scores, t)
        rows.append((t, tp, fp, fn, tn, cost(fn, fp)))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def brute_threshold(table: pd.DataFrame) -> float:
    """Найбільший поріг серед усіх з мінімальною C (не залежить від порядку рядків)."""
    best = table["C"].min()
    return float(table.loc[table["C"] == best, "threshold"].max())


def check_optimizer(out_dir: Path) -> None:
    """Пункт 3: перевірка пошуку порога на наборах A/B/C з умови."""
    test_sets = {
        "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0], 0.4),
        "B": ([0.9, 0.6, 0.3], [0, 0, 0], INF),
        "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0], 0.8),
    }

    print("\nПеревірка пошуку порога на наборах A/B/C:")
    for name, (s_list, y_list, expected) in test_sets.items():
        scores = np.array(s_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64)

        if name == "A":
            assert scores[1] == scores[2] == 0.7 and y[1] != y[2]
        elif name == "B":
            assert np.all(y == 0)
        elif name == "C":
            assert int(np.sum(scores == 0.5)) == 11
            assert int(y[1]) == 1 and int(np.sum(y[2:12] == 0)) == 10

        fast = table_fast(y, scores)
        brute = table_brute(y, scores)

        assert fast.shape == brute.shape, f"Набір {name}: різна кількість кандидатів"
        assert np.array_equal(
            fast["threshold"].to_numpy(), brute["threshold"].to_numpy()
        ), f"Набір {name}: різні кандидати"
        for col in ("TP", "FP", "FN", "TN", "C"):
            assert np.array_equal(
                fast[col].to_numpy(), brute[col].to_numpy()
            ), f"Набір {name}: розбіжність у {col}"

        t_fast, t_brute = pick_threshold(fast), brute_threshold(brute)
        assert t_fast == t_brute == expected, (
            f"Набір {name}: швидкий {t_fast}, перебір {t_brute}, очікувалось {expected}"
        )

        if name == "A":
            tp07, fp07, _, _ = confusion(y, scores, 0.7)
            assert tp07 == 1 and fp07 == 2, (
                "рівні оцінки 0.7 мають обидві перейти поріг одночасно"
            )
        elif name == "B":
            row_inf = fast.loc[fast["threshold"] == INF].iloc[0]
            assert int(row_inf["C"]) == 0
        elif name == "C":
            row08 = fast.loc[fast["threshold"] == 0.8].iloc[0]
            row05 = fast.loc[fast["threshold"] == 0.5].iloc[0]
            assert int(row08["C"]) == int(row05["C"]) == 10
            assert t_fast == 0.8, "tie-break має обрати більший поріг"

        brute.to_csv(out_dir / f"verify_{name}.csv", index=False)
        print(f"\nНабір {name}: t* = {fmt_threshold(t_fast)}  "
              f"(швидкий == перебір == очікуване)")
        shown = brute.copy()
        shown["threshold"] = shown["threshold"].map(fmt_threshold)
        print(shown.to_string(index=False))

    print("\nУсі три перевірки пройдено.")


# ================================================ криві на validation
def plot_curves(
    out_dir: Path,
    y: np.ndarray,
    scores: np.ndarray,
    t_star: float,
) -> tuple[float, float]:
    """PR і ROC на validation з позначками t=0.5 і t*; повертає (AP, ROC-AUC)."""
    base_rate = float(np.mean(y))   

    tp_h, fp_h, fn_h, tn_h = confusion(y, scores, 0.5)
    precision_h = safe_div(tp_h, tp_h + fp_h)
    recall_h = safe_div(tp_h, tp_h + fn_h)

    tp_s, fp_s, fn_s, tn_s = confusion(y, scores, t_star)
    precision_s = safe_div(tp_s, tp_s + fp_s)
    recall_s = safe_div(tp_s, tp_s + fn_s)

    # ---------- PR ----------
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="tab:blue",
             label=f"PR-крива (AP = {ap:.4f})")
    plt.axhline(base_rate, linestyle=":", color="gray",
                label=f"Частка шахрайства = {base_rate:.6f}")
    if precision_h is not None and recall_h is not None:
        plt.scatter([recall_h], [precision_h], color="orange", zorder=5,
                    s=60, label="t = 0.5")
    if precision_s is not None and recall_s is not None:
        plt.scatter([recall_s], [precision_s], color="red", zorder=5,
                    s=60, label=f"t* = {fmt_threshold(t_star)} (мінімум вартості)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR на validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close()

    # ---------- ROC ----------
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = roc_auc_score(y, scores)

    tpr_h = safe_div(tp_h, tp_h + fn_h)
    fpr_h = safe_div(fp_h, fp_h + tn_h)
    tpr_s = safe_div(tp_s, tp_s + fn_s)
    fpr_s = safe_div(fp_s, fp_s + tn_s)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="tab:blue",
             label=f"ROC-AUC = {roc_auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
             label="Випадкове ранжування")
    if tpr_h is not None and fpr_h is not None:
        plt.scatter([fpr_h], [tpr_h], color="orange", zorder=5,
                    s=60, label="t = 0.5")
    if tpr_s is not None and fpr_s is not None:
        plt.scatter([fpr_s], [tpr_s], color="red", zorder=5,
                    s=60, label=f"t* = {fmt_threshold(t_star)} (мінімум вартості)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC на validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()

    return ap, roc_auc


# ======================================= фіксація порога та test
def save_threshold(out_dir: Path, t_star: float) -> None:
    """Зафіксувати t* без округлення (+inf записується як null)."""
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "t_star": None if np.isinf(t_star) else t_star,
                "is_plus_inf": bool(np.isinf(t_star)),
                "t_star_repr": repr(t_star),
            },
            f,
            indent=2,
        )


def load_threshold(out_dir: Path) -> float:
    with open(out_dir / "threshold.json", encoding="utf-8") as f:
        obj = json.load(f)
    return INF if obj["is_plus_inf"] else float(obj["t_star"])


def analyze_test(
    data: dict[str, np.ndarray],
    threshold: float,
    out_dir: Path,
) -> None:
    """Пункт 4: оцінка test зафіксованим t* і розбір помилок."""
    y_test = data["test_y"]
    scores_test = data["test_scores"].astype(np.float64)
    row_ids = data["test_row_ids"]
    amounts = data["test_amount"]

    tp, fp, fn, tn = confusion(y_test, scores_test, threshold)
    total_cost = cost(fn, fp)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    print("\nОцінювання test за зафіксованим порогом:")
    print(f"  Поріг t* = {fmt_threshold(threshold)}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Вартість C = 10*FN + FP = {total_cost}")
    print_metric(
        "Precision", precision,
        "TP + FP = 0 — модель нікого не позначила шахраєм",
    )
    print_metric(
        "Recall", recall,
        "TP + FN = 0 — у test немає шахрайських транзакцій",
    )

    summary = {
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": total_cost,
        "precision": precision, "recall": recall,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)

    predicted = scores_test >= threshold
    parts = []
    for kind, mask in (
        ("FP", (y_test == 0) & predicted),
        ("FN", (y_test == 1) & ~predicted),
    ):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(row_ids[idx], kind="stable")][:5]
        if kind == "FP":
            distance = scores_test[idx] - threshold
        else:
            distance = threshold - scores_test[idx]
        parts.append(
            pd.DataFrame(
                {
                    "csv_row": row_ids[idx],
                    "type": kind,
                    "label": y_test[idx],
                    "score": scores_test[idx],
                    "amount": amounts[idx],
                    "distance": distance,
                }
            )
        )
    errors = pd.concat(parts, ignore_index=True)
    errors.to_csv(out_dir / "test_errors.csv", index=False)

    print("\nПерші 5 FP та перші 5 FN за номером рядка CSV (від 0, без заголовка):")
    if errors.empty:
        print("  Помилок немає.")
    else:
        print(errors.to_string(index=False))

def analyze_test_comparison(
    data: dict[str, np.ndarray],
    threshold: float,
    out_dir: Path,
) -> None:
    """Порівняння правил 0.5 / t* / +inf на test."""
    y = data["test_y"]
    scores = data["test_scores"].astype(np.float64)

    rows = []
    for name, t in (
        ("Стандартний поріг 0.5", 0.5),
        ("Поріг із validation t*", threshold),
        ("Усі негативні +inf", INF),
    ):
        tp, fp, fn, tn = confusion(y, scores, t)
        rows.append({
            "rule": name,
            "threshold": float(t),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "C": cost(fn, fp),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
        })

    df = pd.DataFrame(
        rows,
        columns=["rule", "threshold", "TP", "FP", "FN", "TN",
                 "C", "precision", "recall"],
    )
    df.to_csv(
        out_dir / "test_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row_star = df.loc[df["rule"] == "Поріг із validation t*"].iloc[0]
    summary = pd.read_csv(out_dir / "test_summary.csv").iloc[0]
    assert int(row_star["TP"]) == int(summary["TP"])
    assert int(row_star["FP"]) == int(summary["FP"])
    assert int(row_star["FN"]) == int(summary["FN"])
    assert int(row_star["TN"]) == int(summary["TN"])
    assert int(row_star["C"]) == int(summary["C"])

    print("\nПорівняння правил на test (лише для звіту; поріг не підбирається):")
    shown = df.copy()
    shown["threshold"] = shown["threshold"].map(fmt_threshold)
    print(shown.to_string(index=False))

# ====================================================== main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Аналіз збережених прогнозів для ЛР2 (пункти 1–4)."
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("results/predictions.npz"),
        help="Шлях до predictions.npz",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("analysis"),
        help="Каталог для графіків, CSV і результатів",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    check_optimizer(args.out)

    data = load_predictions(args.predictions)

    sizes = split_summary(data)
    sizes.to_csv(args.out / "split_summary.csv", index=False)
    print("\nРозміри вибірок:")
    print(sizes.to_string(index=False))

    y_val = data["validation_y"]
    scores_val = data["validation_scores"].astype(np.float64)

    table = table_fast(y_val, scores_val)
    table.to_csv(args.out / "validation_candidates.csv", index=False)
    t_star = pick_threshold(table)

    ap, roc_auc = plot_curves(args.out, y_val, scores_val, t_star)
    print(f"\nValidation: n={len(y_val)}, шахрайських={int(y_val.sum())}")
    print(f"Average Precision (validation): {ap:.6f}")
    print(f"ROC-AUC (validation):           {roc_auc:.6f}")

    rows = []
    for name, t in (
        ("Стандартний поріг 0.5", 0.5),
        ("Мінімум вартості t*", t_star),
        ("Усі негативні +inf", INF),
    ):
        tp, fp, fn, tn = confusion(y_val, scores_val, t)
        rows.append({
            "rule": name,
            "threshold": float(t),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "C": cost(fn, fp),
        })
    comparison = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    comparison.to_csv(
        args.out / "validation_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row = table.loc[table["threshold"] == t_star].iloc[0]
    assert [int(row[c]) for c in ("TP", "FP", "FN", "TN", "C")] == [
        int(comparison.loc[1, c]) for c in ("TP", "FP", "FN", "TN", "C")
    ], "таблиця кандидатів розходиться з прямим застосуванням правила"
    assert comparison.loc[1, "C"] <= comparison.loc[0, "C"]
    assert comparison.loc[1, "C"] <= comparison.loc[2, "C"]

    shown = comparison.copy()
    shown["threshold"] = shown["threshold"].map(fmt_threshold)
    print("\nПорівняння правил на validation:")
    print(shown.to_string(index=False))
    print(f"\nОптимальний поріг t* = {fmt_threshold(t_star)}  "
          f"(повне значення — у threshold.json)")

    save_threshold(args.out, t_star)
    fixed = load_threshold(args.out)
    assert fixed == t_star, "t* змінився при збереженні"

    analyze_test(data, fixed, args.out)
    analyze_test_comparison(data, fixed, args.out)

    print(f"\nРезультати збережено в: {args.out.resolve()}")


if __name__ == "__main__":
    main()