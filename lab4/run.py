import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import onnx, onnxruntime as ort, torch, torchvision
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Шлях до lab3_model.pt")
    ap.add_argument("--split", required=True, help="Шлях до split.csv з ЛР3")
    ap.add_argument("--data", required=True, help="Папка, у якій лежать seg_train і seg_test")
    ap.add_argument("--out-dir", default=".", help="Куди писати .onnx і lab4_results.json")
    ap.add_argument("--checks-only", action="store_true",
                    help="Лише контролі, експорт, перевірка відповідності та діапазони каналів")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp32, prep = out / "lab4_fp32.onnx", out / "lab4_prep.onnx"
    int8, int8_pt = out / "lab4_int8.onnx", out / "lab4_int8_per_tensor.onnx"
    res = {}

    # ---- 0. Контроль поділу ----
    rows = read_split(a.split)
    ok, res["split_control"] = check_split(rows)
    print("\n== Контроль поділу (кількість / сума імен) ==")
    for part, v in res["split_control"].items():
        print(f"{part}: {v['n']} / {v['sum']} (очікувано {v['expected_n']} / {v['expected_sum']}) "
              f"-> {'збігається' if v['ok'] else 'НЕ ЗБІГАЄТЬСЯ'}")
    if not ok:
        raise SystemExit("Поділ не збігається з контролем: виправте ЛР3 до початку ЛР4.")

    model, size, ck = load_torch_model(a.model)
    res["model"] = {"regime": ck["regime"], "seed": int(ck["seed"]), "img_size": size}
    print(f"\nМодель: режим {ck['regime']}, зерно {ck['seed']}, img_size {size}")

    test_rows = part_rows(rows, "test")
    y = labels_of(test_rows)
    print("Завантаження test...")
    Xt = load_images(test_rows, a.data, size)

    # ---- 1. Експорт і відповідність ----
    t0 = time.perf_counter()
    export_onnx(model, fp32, size)
    res["export_seconds"] = time.perf_counter() - t0
    sess_fp = make_session(fp32)
    z_torch, z_fp = torch_logits(model, Xt), ort_logits(sess_fp, Xt)
    cnt = node_counts(fp32)
    res["export_check"] = {
        "max_abs_logit_diff": float(np.abs(z_fp - z_torch).max()),
        "class_agreement": float((z_fp.argmax(1) == z_torch.argmax(1)).mean()),
        "macro_f1_torch": macro_f1(y, z_torch), "macro_f1_onnx_fp32": macro_f1(y, z_fp),
        "n_conv": cnt.get("Conv", 0), "n_batchnorm": cnt.get("BatchNormalization", 0),
        "export_seconds": res["export_seconds"]}
    e = res["export_check"]
    print("\n== Експорт і відповідність PyTorch ↔ ONNX FP32 ==")
    print(f"макс. |Δ логітів| = {e['max_abs_logit_diff']:.3e} (очікується порядку 1e-5, менше за 1e-3)")
    print(f"збіг класів = {e['class_agreement']:.4f}")
    print(f"macro-F1: PyTorch {e['macro_f1_torch']:.4f}, ONNX FP32 {e['macro_f1_onnx_fp32']:.4f}")
    print(f"вузлів Conv = {e['n_conv']}, BatchNormalization = {e['n_batchnorm']}")
    print(f"час експорту = {e['export_seconds']:.2f} с")

    # ---- 2. Калібрувальна вибірка ----
    cal_rows = calibration_rows(rows)
    cok, res["calibration_control"] = check_calibration(cal_rows)
    c = res["calibration_control"]
    print("\n== Калібрувальна вибірка ==")
    print(f"n = {c['n']}, сума імен = {c['sum']} (очікувано 1921470), за класами: {c['classes']}")
    print("Контроль:", "збігається" if cok else "НЕ ЗБІГАЄТЬСЯ")
    if not cok:
        raise SystemExit("Калібрувальна вибірка не збігається з контролем.")

    # ---- діапазони каналів (потрібні для прогнозу) ----
    rng = channel_ranges(fp32)
    res["channel_ranges"] = rng
    print("\n== Діапазони каналів (потензорний крок коштує log2 r біт) ==")
    print(f"згорток: {rng['n_conv']}")
    for t in rng["top3"]:
        print(f"{t['node']}: r = {t['r']:.2f}, log2 r = {t['bits']:.2f} біта")
    print(f"медіана r за згортками = {rng['median_r']:.2f}, log2 = {rng['median_bits']:.2f} біта")

    if a.checks_only:
        (out / "lab4_results_checks.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n--checks-only: зупиняємось перед квантизацією. Запишіть прогноз у README.")
        return

    # ---- квантизація ----
    Xc = load_images(cal_rows, a.data, size)
    t0 = time.perf_counter(); prepare(fp32, prep); res["prep_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter(); quantize(prep, int8, Xc, per_channel=True)
    res["quantize_seconds"] = time.perf_counter() - t0
    quantize(prep, int8_pt, Xc, per_channel=False)
    print(f"\nЧас підготовки графа = {res['prep_seconds']:.2f} с, квантизації (per-channel) = {res['quantize_seconds']:.2f} с")

    # ---- 3. Якість ----
    sess_q, sess_pt = make_session(int8), make_session(int8_pt)
    z_q, z_pt = ort_logits(sess_q, Xt), ort_logits(sess_pt, Xt)
    res["quality"] = {"fp32": quality_row(y, z_fp, z_fp), "int8": quality_row(y, z_q, z_fp),
                      "int8_per_tensor": quality_row(y, z_pt, z_fp)}
    res["paired_int8"] = paired(y, z_fp, z_q)
    res["paired_int8_per_tensor"] = paired(y, z_fp, z_pt)
    p = res["quality"]["fp32"]["accuracy"]
    res["se_single_model"] = float((p * (1 - p) / len(y)) ** 0.5)
    print("\n== Якість на test (1502) ==")
    print("Варіант | macro-F1 | accuracy | Згода з FP32")
    for k, v in res["quality"].items():
        print(f"{k} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} | {v['agreement']:.4f}")
    for k in ["paired_int8", "paired_int8_per_tensor"]:
        q = res[k]
        print(f"{k}: b={q['b']}, c={q['c']}, Δacc={q['delta_acc']:+.4f} ± {q['se_paired']:.4f}, "
              f"відрізняється від шуму: {q['differs_from_noise']} {q['direction'] or ''}")
    print(f"SE частки правильних відповідей однієї моделі = {res['se_single_model']:.4f}")

    # ---- 5. Затримка ----
    print("\nВимірювання затримки (ноутбук від мережі, важкі програми закриті)...")
    x1 = np.ascontiguousarray(Xt[0:1])
    lat = measure_latency({"fp32": sess_fp, "int8": sess_q}, x1)
    sz = {"fp32": fp32.stat().st_size / 1e6, "int8": int8.stat().st_size / 1e6}
    res["latency"] = {k: {kk: vv for kk, vv in v.items()} for k, v in lat.items()}
    res["file_size_mb"] = sz
    res["speedup_p50"] = lat["fp32"]["p50"] / lat["int8"]["p50"]
    res["speedup_p95"] = lat["fp32"]["p95"] / lat["int8"]["p95"]
    res["size_ratio"] = sz["fp32"] / sz["int8"]
    res["latency_measured_by_rule"] = not intervals_overlap(
        lat["fp32"]["p50_rounds"], lat["int8"]["p50_rounds"])
    print("Варіант | Файл, МБ | p50, мс | p95, мс | p50 за раундами, мс")
    for k in ["fp32", "int8"]:
        v = lat[k]
        print(f"{k} | {sz[k]:.2f} | {v['p50']:.2f} | {v['p95']:.2f} | "
              f"{[round(t, 2) for t in v['p50_rounds']]}")
    print(f"Прискорення p50 = {res['speedup_p50']:.2f}, p95 = {res['speedup_p95']:.2f}, "
          f"зменшення файлу = {res['size_ratio']:.2f}")
    print("Зміна затримки виміряна за правилом (відрізки p50 за раундами не перетинаються):",
          res["latency_measured_by_rule"])

    res["environment"] = {
        "cpu": cpu_name(), "os": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "intra_op_num_threads": 1, "inter_op_num_threads": 1, "batch_size": 1,
        "input_shape": [1, 3, size, size]}
    (out / "lab4_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nЗбережено lab4_results.json")


if __name__ == "__main__":
    main()