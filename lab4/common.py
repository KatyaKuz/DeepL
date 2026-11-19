import csv, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms as T
from torchvision.models import resnet18
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quant_pre_process,
                                      quantize_static)

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# контрольні числа з завдання: (кількість, сума числових імен файлів)
EXPECTED_PARTS = {"train": (1200, 11993084), "test": (1502, 33351444)}
CAL_EXPECTED = {"n": 200, "sum": 1921470,
                "classes": {"buildings": 40, "forest": 35, "glacier": 36,
                            "mountain": 33, "sea": 29, "street": 27}}


# ---------------- поділ і дані ----------------
def read_split(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def part_rows(rows, part):
    return [r for r in rows if r["part"] == part]


def file_number(name):
    return int(name.split(".")[0])


def check_split(rows):
    """Порівняння частин train і test із рядком «Разом» контрольної таблиці ЛР3."""
    info, ok = {}, True
    for part, (n_exp, s_exp) in EXPECTED_PARTS.items():
        pr = part_rows(rows, part)
        n, s = len(pr), sum(file_number(r["file"]) for r in pr)
        good = (n, s) == (n_exp, s_exp)
        ok &= good
        info[part] = {"n": n, "sum": s, "expected_n": n_exp, "expected_sum": s_exp, "ok": good}
    return ok, info


def calibration_rows(rows):
    """200 рядків train: сортування імен як рядків, потім permutation(1200)[:200] з seed 0."""
    train = sorted(part_rows(rows, "train"), key=lambda r: r["file"])
    idx = np.random.default_rng(0).permutation(1200)[:200]
    return [train[i] for i in idx]


def check_calibration(cal_rows):
    counts = {c: sum(r["label"] == c for r in cal_rows) for c in CLASSES}
    s = sum(file_number(r["file"]) for r in cal_rows)
    ok = (len(cal_rows) == CAL_EXPECTED["n"] and s == CAL_EXPECTED["sum"]
          and counts == CAL_EXPECTED["classes"])
    return ok, {"n": len(cal_rows), "sum": s, "classes": counts, "ok": ok}


def eval_transform(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_images(rows, root, size):
    """Масив float32 форми (N, 3, size, size) у порядку рядків."""
    tf = eval_transform(size)
    out = np.empty((len(rows), 3, size, size), dtype=np.float32)
    for i, r in enumerate(rows):
        sub = "seg_train" if r["part"] == "train" else "seg_test"
        img = Image.open(Path(root) / sub / sub / r["label"] / r["file"]).convert("RGB")
        out[i] = tf(img).numpy()
    return out


def labels_of(rows):
    return np.array([CLASSES.index(r["label"]) for r in rows])


# ---------------- модель PyTorch ----------------
def load_torch_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if list(ck["classes"]) != CLASSES:
        raise SystemExit(f"Порядок класів у файлі моделі {ck['classes']} не збігається з ЛР3")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, int(ck["img_size"]), ck


def torch_logits(model, X, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


# ---------------- ONNX ----------------
def export_onnx(model, path, size):
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, dummy, str(path), dynamo=False, opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})


def make_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def ort_logits(sess, X, bs=64):
    return np.concatenate([sess.run(None, {"input": X[i:i + bs]})[0] for i in range(0, len(X), bs)])


def node_counts(path):
    m = onnx.load(str(path))
    cnt = {}
    for n in m.graph.node:
        cnt[n.op_type] = cnt.get(n.op_type, 0) + 1
    return cnt


class CalibReader(CalibrationDataReader):
    """Калібрувальні зображення подаються по одному: форма (1, 3, H, W)."""
    def __init__(self, X):
        self.it = iter([{"input": X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)


def quantize(prep_path, out_path, X_cal, per_channel):
    quantize_static(str(prep_path), str(out_path), CalibReader(X_cal),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                    calibrate_method=CalibrationMethod.MinMax)


def prepare(fp32_path, prep_path):
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)


# ---------------- діапазони каналів ----------------
def channel_ranges(fp32_path):
    """Для кожного Conv: alpha_c = max|w| за елементами вихідного каналу, r = max alpha / median alpha."""
    m = onnx.load(str(fp32_path))
    inits = {i.name: i for i in m.graph.initializer}
    res = []
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        w = numpy_helper.to_array(inits[n.input[1]])          # (Cout, Cin, kH, kW)
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        res.append({"node": n.name, "shape": list(w.shape), "r": r, "bits": float(np.log2(r))})
    rs = np.array([x["r"] for x in res])
    top3 = sorted(res, key=lambda x: -x["r"])[:3]
    med = float(np.median(rs))
    return {"n_conv": len(res), "top3": top3, "median_r": med,
            "median_bits": float(np.log2(med)), "all": res}


# ---------------- якість ----------------
def macro_f1(y, z):
    return float(f1_score(y, z.argmax(1), average="macro"))


def quality_row(y, z, z_fp):
    pred, pred_fp = z.argmax(1), z_fp.argmax(1)
    return {"macro_f1": macro_f1(y, z), "accuracy": float((pred == y).mean()),
            "agreement": float((pred == pred_fp).mean())}


def paired(y, z_fp, z_q):
    cf, cq = z_fp.argmax(1) == y, z_q.argmax(1) == y
    b = int((cq & ~cf).sum())          # INT8 правильно, FP32 помиляється
    c = int((~cq & cf).sum())          # FP32 правильно, INT8 помиляється
    n = len(y)
    s = (b + c) ** 0.5
    differs = abs(b - c) > 2 * s
    direction = None
    if differs:
        direction = "INT8 краща" if b > c else "INT8 гірша"
    return {"b": b, "c": c, "delta_acc": (b - c) / n, "se_paired": s / n,
            "differs_from_noise": bool(differs), "direction": direction}


# ---------------- затримка ----------------
def measure_latency(sessions, x, rounds=3, warmup=20, n=100):
    """sessions: {'fp32': sess, 'int8': sess}. У кожному раунді спершу fp32, потім int8."""
    feed = {"input": x}
    per_round = {k: [] for k in sessions}
    for _ in range(rounds):
        for name, sess in sessions.items():
            for _ in range(warmup):
                sess.run(None, feed)
            t = []
            for _ in range(n):
                t0 = time.perf_counter()
                sess.run(None, feed)
                t1 = time.perf_counter()
                t.append((t1 - t0) * 1000.0)
            per_round[name].append(t)
    out = {}
    for name, rr in per_round.items():
        allv = np.concatenate(rr)
        out[name] = {"p50": float(np.percentile(allv, 50)), "p95": float(np.percentile(allv, 95)),
                     "p50_rounds": [float(np.percentile(r, 50)) for r in rr],
                     "measurements_ms": [list(map(float, r)) for r in rr]}
    return out


def intervals_overlap(a, b):
    return not (max(a) < min(b) or max(b) < min(a))


# ---------------- середовище ----------------
def cpu_name():
    try:
        r = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True).strip()
        if r:
            return r
    except Exception:
        pass
    return platform.processor()