
# Main.py


# ------------------------------------------------ Block 1 : Imports ------------------------------------------------#
import os
import sys
import threading
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pygame

# Try to import sounddevice; if unavailable, the app still runs (CSV mode only)
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

# Tk for file dialog (hidden root window)
import tkinter as tk
from tkinter import filedialog, messagebox

import re


# ------------------------------------------------ Block 1 : End ------------------------------------------------#

# --------------------------------------- Block 2 : Config & constants ------------------------------------------#

WIDTH, HEIGHT = 1200, 700
FPS = 60

# Plot areas (split view)
TOP_PAD = 80
BOTTOM_PAD = 60
MID_GAP = 10
PLOT_TOP_H = (HEIGHT - TOP_PAD - BOTTOM_PAD - MID_GAP) // 2
PLOT_BOT_Y = TOP_PAD + PLOT_TOP_H + MID_GAP
BG = (12, 14, 20)
GRID = (36, 40, 52)
AXIS = (80, 86, 100)
WAVE = (0, 200, 255)
SPEC = (255, 170, 0)
TEXT = (230, 231, 235)
MUTED = (150, 155, 165)
ACCENT = (120, 200, 120)
RED = (230, 90, 90)
YELLOW = (250, 220, 120)

SAMPLE_RATE = 48000   # audio capture Fs
AUDIO_BLOCK = 2048    # block size samples
RING_SEC = 2.0
RING_LEN = int(SAMPLE_RATE * RING_SEC)

# --------------------------------------- Block 2 : End ------------------------------------------#


# --------------------------------------- Block 3 : Data structures ------------------------------------------#

@dataclass
class StandardSignal:
    time: np.ndarray            # seconds
    amplitude: np.ndarray       # volts (or normalized amplitude)
    sampling_rate: float        # Hz
    source: str                 # "csv" or "audio"


@dataclass
class AnalysisMetrics:
    f0_hz: Optional[float]
    thd_percent: Optional[float]
    vpp: Optional[float]
    vrms: Optional[float]
    dc: Optional[float]
    duty_percent: Optional[float]           # N/A -> None
    triangle_skew_percent: Optional[float]  # N/A -> None
    mode: str                               # "auto","sine","square","triangle"
    detected_label: str                     # e.g., "Sine (82%)" or "Square (55%)"


# --------------------------------------- Block 3 : End ------------------------------------------#

# -

# -

# --------------------------------------- Block 4 : CSV Import ------------------------------------------#

# Regex to match numbers Examples matched: '12', '-8.5', '+3,1415', '1.2e-3', '2,5E6'
DECIMAL_SEP_RE = re.compile(r"[-+]?\d{1,3}(?:[\.,]\d+)?(?:[eE][-+]?\d+)?")

# Dictionary to scale units to normalised SI variations
_UNIT_SCALE = {
    "s": 1.0, "sec": 1.0, "secs": 1.0,
    "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9,
    "v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6,
}


def _read_text_head(path: str, max_lines: int = 400) -> List[str]:
    """Read up to max_lines from file; if max_lines is None or <=0, read entire file."""
    lines: List[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:  # utf-8-sig strips BOMs
        if max_lines is None or max_lines <= 0:
            for line in f:
                lines.append(line.rstrip("\n\r"))
            return lines
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line.rstrip("\n\r"))
    return lines

def _sniff_delimiter_and_decimal(lines: List[str]) -> Tuple[str, str]:
    """
    Returns (delimiter, decimal) where decimal is '.' or ','
    """
    candidate_delims = [",", ";", "\t"]
    # look for a line that "looks numeric"
    nums_line = None
    for ln in lines:
        if sum(ch.isdigit() for ch in ln) >= 4:
            nums_line = ln
            break
    if nums_line is None:
        return ",", "."  # default

    # choose delimiter with the highest split producing 2+ fields
    best_delim = ","
    best_count = -1
    for d in candidate_delims:
        c = nums_line.count(d)
        if c > best_count:
            best_delim, best_count = d, c

    # decimal separator: if delimiter is ';' and we see commas in numbers -> decimal=','
    dec = "."
    if best_delim == ";":
        if re.search(r"\d,\d", nums_line):
            dec = ","
    else:
        # try to detect european style even with comma delimiter (rare)
        # prefer '.' unless we see many patterns like 1,23 and almost no dots.
        comma_nums = len(re.findall(r"\d,\d", nums_line))
        dot_nums = len(re.findall(r"\d\.\d", nums_line))
        if comma_nums > dot_nums * 3:
            dec = ","
    return best_delim, dec


def _strip_units(token: str) -> Tuple[str, float]:
    """
    Extract unit in parentheses or suffix: e.g. 'Time (ms)' -> ('Time', 1e-3)
    Returns (clean_label, scale)
    """
    t = token.strip().lower()
    # parentheses
    m = re.search(r"\(([^)]+)\)", t)
    unit = None
    if m:
        unit = m.group(1).strip().lower()
    else:
        # suffix after space, e.g. "Voltage V" or "CH1[V]"
        m2 = re.search(r"\[?([a-zµu]{1,2}v|[munµ]s|ms|us|ns|s)\]?$", t)
        if m2:
            unit = m2.group(1).strip().lower()
    scale = _UNIT_SCALE.get(unit, 1.0) if unit else 1.0
    # clean label (remove brackets/units)
    label = re.sub(r"[\[\(].*?[\]\)]", "", token, flags=re.IGNORECASE).strip()
    return label, scale


def _parse_header_for_metadata(lines: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to find sample interval (seconds) or sample rate (Hz) in header.
    Returns (dt, fs) where either can be None.
    """
    head = "\n".join(lines[:120]).lower()
    # sample interval / delta t
    m = re.search(r"(sample\s*interval|xinc|time\s*increment|dt|delta\s*x)\s*[,:\t ]\s*([-+.\deE]+)\s*([munµ]?s|ms|us|ns|s)?", head)
    if m:
        val = float(m.group(2).replace(",", "."))
        unit = (m.group(3) or "s").lower()
        scale = _UNIT_SCALE.get(unit, 1.0)
        return val * scale, None
    # sample rate
    m = re.search(r"(sample\s*rate|sampling\s*rate|sa/s)\s*[,:\t ]\s*([-+.\deE]+)\s*(hz|khz|mhz|ghz|sa/s)?", head)
    if m:
        val = float(m.group(2).replace(",", "."))
        unit = (m.group(3) or "hz").lower()
        mult = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "sa/s": 1.0}.get(unit, 1.0)
        return None, val * mult
    return None, None


def _find_data_start_and_headers(lines: List[str], delim: str) -> Tuple[int, Optional[List[str]]]:
    """
    Find the first data line index. If a header row with column names exists just before, return those names.
    """
    # Consider a line "data-like" if splitting yields 1-4 fields with mostly numeric tokens
    def looks_data(ln: str) -> bool:
        parts = [p.strip() for p in ln.split(delim)]
        if len(parts) < 1:
            return False
        numericish = 0
        for p in parts[:4]:
            if DECIMAL_SEP_RE.match(p.replace(" ", "").replace("\t", "")):
                numericish += 1
        return numericish >= max(1, min(2, len(parts)))

    header_names = None
    for i, ln in enumerate(lines):
        if looks_data(ln):
            # Check if previous non-empty line looks like column headers
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0:
                prev = [p.strip() for p in lines[j].split(delim)]
                # headers if tokens contain non-numeric labels
                if any(re.search(r"[A-Za-z]", p) for p in prev):
                    header_names = prev
            return i, header_names
    # fallback: assume first non-empty line
    for i, ln in enumerate(lines):
        if ln.strip():
            return i, None
    return 0, None


def _tok_to_float(tok: str) -> Optional[float]:
    tok = tok.strip().strip('"').strip("'")
    if not tok:
        return None
    # Decimal comma -> dot (when token looks numeric)
    if re.search(r"\d,\d", tok):
        tok = re.sub(r"(?<=\d),(?=\d)", ".", tok)
    try:
        return float(tok)
    except Exception:
        return None


def _collect_numeric_rows(lines: List[str], delim: str) -> List[List[float]]:
    """Traverse lines and collect rows with 1+ numeric tokens after splitting by delim."""
    rows: List[List[float]] = []
    for ln in lines:
        if not ln or not ln.strip():
            continue
        parts = [p for p in ln.split(delim)]
        nums: List[float] = []
        for p in parts:
            v = _tok_to_float(p)
            if v is not None:
                nums.append(v)
        if len(nums) >= 1:
            rows.append(nums)
    return rows


def _extract_longest_sane_run(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the longest contiguous run where dt is positive and close to the median.
    This drops footer/junk rows that sneak in with huge time jumps.
    """
    if t.size < 3:
        return t, y

    # Sort by time, drop exact duplicates first
    ord_idx = np.argsort(t)
    t = t[ord_idx]; y = y[ord_idx]
    dt = np.diff(t)
    finite = np.isfinite(t) & np.isfinite(y)
    if not np.all(finite):
        t = t[finite]; y = y[finite]
        if t.size < 3: return t, y
        dt = np.diff(t)

    # Median (robust) step
    dt_pos = dt[dt > 0]
    if dt_pos.size == 0:
        return t[:1], y[:1]

    dt_med = float(np.median(dt_pos))
    if not (dt_med > 0):
        return t[:1], y[:1]

    # A "sane" step: 0 < dt < k * dt_med (k is generous)
    k = 20.0
    good = (dt > 0) & (dt < k * dt_med)

    # Find the longest contiguous True run in 'good'
    best_len = 0; best_i0 = 0
    i = 0
    n = good.size
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < n and good[j]:
            j += 1
        if (j - i) > best_len:
            best_len = (j - i)
            best_i0 = i
        i = j

    # If nothing found, fall back to the earliest portion where dt>0
    if best_len <= 0:
        # keep first small block of positive steps
        pos = np.where(dt > 0)[0]
        if pos.size:
            i0 = int(pos[0]); i1 = int(min(pos[0] + 256, t.size - 1))
            return t[i0:i1+1], y[i0:i1+1]
        return t[: min(2, t.size)], y[: min(2, y.size)]

    # The run in dt indexes maps to samples [i0 : i1+1]
    i0 = best_i0
    i1 = best_i0 + best_len  # inclusive on the right in samples
    t_run = t[i0:i1+1]
    y_run = y[i0:i1+1]

    # Rebase time to start at 0 for neatness
    t_run = t_run - t_run[0]
    return t_run, y_run



def load_scope_csv_robust(path: str) -> Optional[StandardSignal]:
    try:
        # Read ENTIRE file so we don't miss late headers/data
        lines = _read_text_head(path, max_lines=None)
        if not lines:
            raise ValueError("Empty file")

        # 1) Sniff delimiter and decimal
        delim, dec = _sniff_delimiter_and_decimal(lines)

        # 2) Normalize to comma delimiter for a simple split pass
        norm_lines = [ln.replace(delim, ",") if delim != "," else ln for ln in lines]
        # If decimal comma, normalize numeric commas to dots (keep commas as delimiters)
        if dec == ",":
            norm_lines = [re.sub(r"(?<=\d),(?=\d)", ".", ln) for ln in norm_lines]

        # 3) Find first data-ish line and optional header names (best-effort)
        data_start, header_names = _find_data_start_and_headers(norm_lines, ",")
        raw_data_lines = norm_lines[data_start:]
        if not raw_data_lines or all(not ln.strip() for ln in raw_data_lines):
            raise ValueError("No data rows detected after header.")

        # 4) Manually collect numeric rows (robust to junk text)
        rows = _collect_numeric_rows(raw_data_lines, ",")
        if not rows:
            raise ValueError("Found header text but no numeric rows to parse.")

        # 5) Decide column layout
        ncols = max(len(r) for r in rows)
        # Build arrays with 1 or 2 columns (truncate extras, skip short rows)
        if ncols >= 2:
            # Use the first two numeric columns per row
            t_vals, y_vals = [], []
            for r in rows:
                if len(r) >= 2:
                    t_vals.append(r[0]);
                    y_vals.append(r[1])
            if len(t_vals) < 2:
                raise ValueError("Insufficient two-column numeric data.")

            t = np.asarray(t_vals, dtype=float)
            y = np.asarray(y_vals, dtype=float)

            # Unit scaling if header suggests units
            time_scale = 1.0;
            amp_scale = 1.0
            if header_names:
                names = [h.strip() for h in header_names]
                if len(names) >= 1:
                    _, time_scale = _strip_units(names[0])
                if len(names) >= 2:
                    _, amp_scale = _strip_units(names[1])
            t = t * time_scale
            y = y * amp_scale

            # Filter to finite values
            mask = np.isfinite(t) & np.isfinite(y)
            t, y = t[mask], y[mask]

            # >>> Robust cleanup: keep only the longest sane run of samples <<<
            t, y = _extract_longest_sane_run(t, y)

            if t.size < 2:
                raise ValueError("After cleaning, not enough time samples remained.")

            # Infer Fs from time; fall back to SAMPLE_RATE if degenerate
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            fs = (1.0 / np.median(dt)) if dt.size else float(SAMPLE_RATE)

            # TODO remove code later

            # Debuging Code
            print(f"[CSV] rows={len(rows)}  ncols={ncols}  header_names={header_names}")
            print(f"[CSV] t[0:3]={t[:3] if 't' in locals() else '—'}  t[-3:]={t[-3:] if 't' in locals() else '—'}")
            print(f"[CSV] y[0:3]={y[:3] if 'y' in locals() else '—'}  y[-3:]={y[-3:] if 'y' in locals() else '—'}")
            if 't' in locals() and len(t) > 1:
                dt = np.diff(t)
                print(
                    f"[CSV] dt>0 count={np.sum(dt > 0)}  dt_med={np.median(dt[dt > 0]) if np.any(dt > 0) else '—'}")
            # End of debuging


            return StandardSignal(time=t, amplitude=y, sampling_rate=fs, source="csv")


        else:
            # -------- amplitude-only CSV --------
            y = np.asarray([r[0] for r in rows], dtype=float)
            y = y[np.isfinite(y)]
            if y.size < 2:
                raise ValueError("Amplitude-only data has too few numeric points.")

            # Try header for dt/fs; else, graceful default to SAMPLE_RATE
            dt_meta, fs_meta = _parse_header_for_metadata(lines)
            if dt_meta is None and fs_meta is None:
                fs = float(SAMPLE_RATE)  # fallback
            else:
                fs = float(fs_meta) if fs_meta else float(1.0 / dt_meta)

            t = np.arange(len(y), dtype=float) / fs

            # TODO remove code later

            # Debuging Code
            print(f"[CSV] rows={len(rows)}  ncols={ncols}  header_names={header_names}")
            print(f"[CSV] t[0:3]={t[:3] if 't' in locals() else '—'}  t[-3:]={t[-3:] if 't' in locals() else '—'}")
            print(f"[CSV] y[0:3]={y[:3] if 'y' in locals() else '—'}  y[-3:]={y[-3:] if 'y' in locals() else '—'}")
            if 't' in locals() and len(t) > 1:
                dt = np.diff(t)
                print(f"[CSV] dt>0 count={np.sum(dt > 0)}  dt_med={np.median(dt[dt > 0]) if np.any(dt > 0) else '—'}")
            # End of debuging


            return StandardSignal(time=t, amplitude=y, sampling_rate=fs, source="csv")

    except Exception as e:
        # Surface a clear, actionable message — the UI handler will also catch/report.
        messagebox.showerror("CSV Load Error", f"{os.path.basename(path)}\n{e}")
        return None


# --------------------------------------- Block 4 : End ------------------------------------------#


# -


# -


# --------------------------------------- Block 5 : Audio In ------------------------------------------#

class AudioStream:
    """Continuous audio capture -> ring buffer"""
    def __init__(self, fs=SAMPLE_RATE, channels=1):
        self.fs = fs
        self.channels = channels
        self.lock = threading.Lock()
        self.ring = np.zeros(RING_LEN, dtype=np.float32)
        self.idx = 0
        self.running = False
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # You can print status if debugging
            pass
        samples = indata[:, 0].copy() if self.channels > 1 else indata.copy().reshape(-1)
        with self.lock:
            n = len(samples)
            end = self.idx + n
            if end <= RING_LEN:
                self.ring[self.idx:end] = samples
            else:
                first = RING_LEN - self.idx
                self.ring[self.idx:] = samples[:first]
                self.ring[:end % RING_LEN] = samples[first:]
            self.idx = end % RING_LEN

    def start(self):
        if not HAVE_SD:
            messagebox.showwarning("Audio", "sounddevice not installed; live capture disabled.")
            return
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.fs,
            blocksize=AUDIO_BLOCK,
            callback=self._callback,
            dtype='float32'
        )
        self.stream.start()

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

    def snapshot(self, window_sec: float = 0.2) -> Optional[StandardSignal]:
        if not self.running:
            return None
        n = int(self.fs * window_sec)
        if n <= 0 or n > RING_LEN:
            n = min(AUDIO_BLOCK * 4, RING_LEN)
        with self.lock:
            end = self.idx
            start = (end - n) % RING_LEN
            if start < end:
                y = self.ring[start:end].copy()
            else:
                y = np.concatenate([self.ring[start:], self.ring[:end]]).copy()
        t = np.arange(len(y)) / float(self.fs)
        return StandardSignal(time=t, amplitude=y.astype(np.float64), sampling_rate=float(self.fs), source="audio")

# --------------------------------------- Block 5 : End ------------------------------------------#


# -


# -


# --------------------------------------- Block 6 : Analysis utilities ------------------------------------------#

def compute_fft(signal: StandardSignal) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal.amplitude, dtype=np.float64)
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    # Hann window -> better spectral estimates
    w = np.hanning(n)
    yf = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(n, d=1.0 / signal.sampling_rate)
    mag = np.abs(yf) * 2.0 / np.sum(w)  # amplitude scaling to approx. volts
    return freqs, mag


def estimate_fundamental(freqs: np.ndarray, mag: np.ndarray) -> Optional[float]:
    if len(freqs) < 3:
        return None
    # ignore DC bin
    mag = mag.copy()
    mag[0] = 0.0
    idx = np.argmax(mag)
    if mag[idx] <= 0:
        return None
    return float(freqs[idx])


# Computes basic wave form statistics: Vpp Vrms and DC offset
def compute_basic_levels(y: np.ndarray) -> Tuple[float, float, float]:
    if len(y) == 0:
        return (None, None, None)
    vpp = float(np.nanmax(y) - np.nanmin(y))
    vrms = float(np.sqrt(np.mean(np.square(y - np.mean(y)))))
    dc = float(np.mean(y))
    return vpp, vrms, dc


# Calculates THD
def compute_thd(freqs: np.ndarray, mag: np.ndarray, f0: Optional[float], max_harmonics: int = 10) -> Optional[float]:
    if f0 is None or f0 <= 0 or len(freqs) == 0:
        return None
    fs_bin = freqs[1] - freqs[0]
    def bin_for(f): return int(round(f / fs_bin))
    h1_bin = bin_for(f0)
    if h1_bin <= 0 or h1_bin >= len(mag):
        return None
    h1 = mag[h1_bin]
    if h1 <= 1e-12:
        return None
    harm_energy = 0.0
    for k in range(2, max_harmonics + 1):
        b = bin_for(k * f0)
        if b < len(mag):
            harm_energy += mag[b] ** 2
    thd = math.sqrt(harm_energy) / h1
    return float(thd * 100.0)


# Calculates Duty cycle
def compute_duty_cycle(y: np.ndarray) -> Optional[float]:
    if len(y) < 4:
        return None
    # Robust threshold using median of upper/lower halves
    median = np.median(y)
    hi = y[y >= median]
    lo = y[y < median]
    if len(hi) < 5 or len(lo) < 5:
        return None
    thresh = 0.5 * (np.mean(hi) + np.mean(lo))

    # Find cycles via zero-crossing on a high-pass-ish version (or use threshold crossings)
    above = y >= thresh
    # Identify rising edges (False->True)
    edges = np.where(np.diff(above.astype(np.int8)) == 1)[0]
    if len(edges) < 2:
        return None

    highs = []
    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i + 1]
        seg = above[start:end]
        highs.append(np.sum(seg) / len(seg))
    if not highs:
        return None
    duty = float(np.mean(highs) * 100.0)
    # sanity check: require bi modality-ish spread
    if duty < 1.0 or duty > 99.0:
        return None
    return duty


# Calculates Skew of triangle
def compute_triangle_skew(y: np.ndarray) -> Optional[float]:
    if len(y) < 8:
        return None
    # Approx: detect peaks & troughs and compare rise vs. fall durations
    # Smooth a little
    ys = y if len(y) < 1024 else moving_avg(y, 5)
    # derivative sign
    d = np.diff(ys)
    sign = np.sign(d)
    # zero-crossings of derivative -> peaks/troughs
    z = np.where(np.diff(sign) != 0)[0]
    if len(z) < 3:
        return None
    # pick consecutive segments
    periods = []
    for i in range(len(z) - 2):
        a, b, c = z[i], z[i + 1], z[i + 2]
        rise = b - a
        fall = c - b
        if rise + fall > 0:
            skew = (rise - fall) / (rise + fall)
            periods.append(skew)
    if not periods:
        return None
    return float(np.mean(periods) * 100.0)


# Calculates a moving average
def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    c = np.convolve(x, np.ones(n) / n, mode='same')
    return c


# Determines the type of waveform
def classify_waveform(freqs: np.ndarray, mag: np.ndarray, y: np.ndarray) -> Tuple[str, str]:
    """
    Simple, transparent rules. Returns (label, confidence_str).
    """
    if len(freqs) < 4 or len(y) < 8:
        return "Unknown", "0%"

    # compute harmonic ratios
    f0 = estimate_fundamental(freqs, mag)
    if not f0:
        return "Unknown", "0%"

    fs_bin = freqs[1] - freqs[0]
    bin_for = lambda f: int(round(f / fs_bin))
    h1 = mag[bin_for(f0)] if bin_for(f0) < len(mag) else 0.0
    h2 = mag[bin_for(2 * f0)] if bin_for(2 * f0) < len(mag) else 0.0
    h3 = mag[bin_for(3 * f0)] if bin_for(3 * f0) < len(mag) else 0.0
    h4 = mag[bin_for(4 * f0)] if bin_for(4 * f0) < len(mag) else 0.0

    # Bimodality for square-ish
    median = np.median(y)
    hi = y[y >= median]
    lo = y[y < median]
    bimodal = (len(hi) > 5 and len(lo) > 5 and
               (np.std(hi) < 0.5 * np.std(y) and np.std(lo) < 0.5 * np.std(y)))

    # Heuristics
    # Sine: h1 dominates, overtones tiny
    if h1 > 0 and (h2 + h3 + h4) < 0.2 * h1:
        conf = 0.7 if not bimodal else 0.55
        return "Sine", f"{int(conf*100)}%"

    # Square: strong odd harmonics, bimodal amplitude
    if h3 > 0.2 * h1 and h2 < 0.2 * h1 and bimodal:
        return "Square", "75%"

    # Triangle: h2 small, h3 present but decays fast, ramps visible (low derivative variance)
    deriv = np.diff(y)
    if h3 > 0.1 * h1 and h2 < 0.1 * h1 and np.std(deriv) < 0.8 * np.std(y):
        return "Triangle", "65%"

    return "Unknown", "50%"
# --------------------------------------- Block 6 : End ------------------------------------------#


# -


# -


# --------------------------------------- Block 7 : Calibration ------------------------------------------#

@dataclass
class ViewScale:
    # Scales for drawing
    volts_per_div: float
    secs_per_div: float
    v_offset: float        # vertical center offset (volts)
    t_start: float = 0.0   # left-edge time for the visible window (seconds)


@dataclass
class SpectrumScale:
    hz_per_div: float     # horizontal frequency scale
    f_start: float        # left-edge frequency (Hz)


def auto_calibrate(
    signal: StandardSignal,
    plot_rect: pygame.Rect,
    cycles_target: int = 4,
    fit_margin: float = 0.90
) -> ViewScale:
    """
    Choose scales so ~cycles_target cycles are visible horizontally and the vertical
    range fills ~fit_margin of the plot height. Ensures at least a minimum number
    of samples are visible and clamps to the available time span. Sets t_start to
    show the last portion of the signal.
    """
    y = np.asarray(signal.amplitude, dtype=float)
    t = np.asarray(signal.time, dtype=float)
    if y.size < 2 or t.size < 2:
        return ViewScale(1.0, 0.01, 0.0, 0.0)

    # --- Vertical fit ---
    y_max = np.nanmax(y); y_min = np.nanmin(y)
    vpp = float(y_max - y_min) if np.isfinite(y_max - y_min) else 1.0
    if vpp <= 1e-12:
        vpp = 1.0
    volts_per_div = (vpp / fit_margin) / 6.0
    v_offset = float((y_max + y_min) * 0.5)

    # --- Horizontal fit ---
    total_span = float(t[-1] - t[0])
    dt_med = float(np.median(np.diff(t))) if t.size > 1 else (1.0 / max(signal.sampling_rate, 1.0))

    # Try to set span from f0; make sure we still see enough samples
    freqs, mag = compute_fft(signal)
    f0 = estimate_fundamental(freqs, mag)
    span_from_cycles = (cycles_target / f0) if (f0 and f0 > 0) else min(total_span, 0.5)

    # Require at least ~200 samples (or fewer if the array is small)
    min_vis_samples = min(max(200, int(0.02 / max(dt_med, 1e-9))), y.size)  # ≥200 or ~20ms worth
    span_from_samples = max(min_vis_samples * dt_med, 5 * dt_med)

    visible_span = max(span_from_cycles, span_from_samples)
    visible_span = min(visible_span, max(total_span, dt_med))

    # Convert span -> secs/div (10 divs across, with margin)
    secs_per_div = max(visible_span / (10.0 * fit_margin), 1e-9)

    # Show the *last* chunk of the trace
    t_start = max(t[0], t[-1] - visible_span)

    return ViewScale(volts_per_div=volts_per_div, secs_per_div=secs_per_div, v_offset=v_offset, t_start=t_start)


def ensure_visible_window(signal: StandardSignal, scale: ViewScale) -> None:
    """
    Keep the 10-division window inside the data. Do NOT touch secs_per_div unless
    the window is empty (<2 samples). This preserves user zoom/pan.
    """
    t = np.asarray(signal.time, dtype=np.float64)
    if t.size < 2:
        return

    t_data0, t_data1 = float(t[0]), float(t[-1])
    # Current window
    span = 10.0 * max(scale.secs_per_div, 1e-12)
    # Clamp t_start so the window lies within the data span
    t0 = float(scale.t_start)
    t0 = max(t_data0, min(t0, t_data1 - span))
    t1 = t0 + span

    # Check how many samples are visible
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    if (i1 - i0) >= 2:
        # Enough samples → keep user's zoom; just write back the clamped start.
        scale.t_start = t0
        return

    # ---- Minimal recovery when the window is empty/thin ----
    # 1) Slide to the tail with the SAME span and try again.
    t0_tail = max(t_data0, t_data1 - span)
    i0 = int(np.searchsorted(t, t0_tail, side="left"))
    i1 = int(np.searchsorted(t, t0_tail + span, side="right"))
    if (i1 - i0) >= 2:
        scale.t_start = t0_tail
        return

    # 2) Still too few → widen just enough to include at least a couple of samples,
    #    but never exceed the full data span. Now it's OK to adjust secs_per_div.
    dt_med = float(np.median(np.diff(t))) if t.size > 1 else (1.0 / max(signal.sampling_rate, 1.0))
    if not (dt_med > 0):
        dt_med = (1.0 / max(signal.sampling_rate, 1.0))

    total_span = max(t_data1 - t_data0, dt_med)
    needed_span = max(5 * dt_med, 2 * dt_med)  # ~a handful of samples
    span = min(max(span, needed_span), total_span)

    scale.secs_per_div = max(span / 10.0, 1e-12)
    scale.t_start = max(t_data0, t_data1 - span)


def auto_calibrate_spectrum(freqs: np.ndarray, fit_margin: float = 0.98) -> SpectrumScale:
    """
    Show the full available band (0..Nyquist) across 10 divisions.
    """
    if freqs.size < 2:
        return SpectrumScale(hz_per_div=1000.0, f_start=0.0)
    f_max = float(freqs[-1])
    span = f_max / max(fit_margin, 1e-6)
    hz_per_div = max(span / 10.0, 1e-3)
    return SpectrumScale(hz_per_div=hz_per_div, f_start=0.0)


def ensure_visible_freq_window(scale: SpectrumScale, f_max: float) -> None:
    """
    Clamp f_start/hz_per_div so the 10-division window is inside [0, f_max].
    """
    f_span = 10.0 * max(scale.hz_per_div, 1e-9)
    if f_span <= 0:
        f_span = 1.0
    scale.hz_per_div = f_span / 10.0  # normalize
    # clamp
    max_start = max(0.0, f_max - f_span)
    scale.f_start = min(max(0.0, scale.f_start), max_start)


# --------------------------------------- Block 7 : End ------------------------------------------#



# -


# -


# --------------------------------------- Block 8 : Drawing ------------------------------------------#


class Button:
    def __init__(self, rect, label, on_click, toggle=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.toggle = toggle
        self.active = False

    def draw(self, surf, font):
        color = (60, 65, 80) if not self.active else (80, 120, 200)
        pygame.draw.rect(surf, color, self.rect, border_radius=10)
        pygame.draw.rect(surf, (90, 95, 110), self.rect, width=2, border_radius=10)
        text = font.render(self.label, True, TEXT)
        surf.blit(text, text.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                self.on_click(self)


def draw_grid(surf, rect: pygame.Rect, x_divs=10, y_divs=6):
    pygame.draw.rect(surf, (22, 25, 35), rect, border_radius=6)
    # grid lines
    for i in range(1, x_divs):
        x = rect.x + int(rect.w * i / x_divs)
        pygame.draw.line(surf, GRID, (x, rect.y), (x, rect.bottom))
    for j in range(1, y_divs):
        y = rect.y + int(rect.h * j / y_divs)
        pygame.draw.line(surf, GRID, (rect.x, y), (rect.right, y))
    pygame.draw.rect(surf, AXIS, rect, width=2, border_radius=6)


def draw_waveform_points(
    surf,
    rect: pygame.Rect,
    signal: Optional[StandardSignal],
    scale: ViewScale,
    font_small,
    point_color=WAVE,
    radius: int = 2
):
    """
    Minimal plotting: render raw samples as individual points (no interpolation, no lines).
    Uses the current ViewScale (secs/div, volts/div, v_offset, t_start)
    and only draws samples that fall inside the visible 10-division time window.
    """
    draw_grid(surf, rect)
    if signal is None or signal.time.size < 2:
        return

    # Numpy aliases
    t = np.asarray(signal.time, dtype=np.float64)
    y = np.asarray(signal.amplitude, dtype=np.float64)

    # --- visible window in time ---
    span = 10.0 * max(scale.secs_per_div, 1e-12)
    t0 = float(scale.t_start)
    t1 = t0 + span

    # Intersect with available data; if empty, bail
    t0 = max(t0, float(t[0]))
    t1 = min(t1, float(t[-1]))
    if not (t1 > t0):
        return

    # --- choose only samples inside [t0, t1] ---
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    if (i1 - i0) <= 0:
        return
    t_vis = t[i0:i1]
    y_vis = y[i0:i1]

    # --- map volts → pixels ---
    v_off = float(scale.v_offset)
    vpix_per_div = rect.h / 6.0
    px_per_v = vpix_per_div / max(scale.volts_per_div, 1e-12)

    # --- map time → pixels ---
    # Note: use (t - t0)/span to place points left→right in the rect
    xs = rect.x + ((t_vis - t0) / max(span, 1e-12)) * rect.w
    ys = rect.centery - (y_vis - v_off) * px_per_v

    # Clip and convert to integers once for speed
    xs = np.clip(xs, rect.x, rect.right - 1).astype(int)
    ys = np.clip(ys, rect.y, rect.bottom - 1).astype(int)

    # To avoid overdraw with giant files, thin points if 1M+ samples
    max_points = rect.w * 3  # ~3 points per column worst-case
    if xs.size > max_points:
        step = int(np.ceil(xs.size / max_points))
        xs = xs[::step]; ys = ys[::step]

    # Draw points
    if radius <= 1:
        # per-pixel plot
        for x, ypix in zip(xs, ys):
            surf.set_at((int(x), int(ypix)), point_color)
    else:
        # tiny filled circles (nicer)
        for x, ypix in zip(xs, ys):
            pygame.draw.circle(surf, point_color, (int(x), int(ypix)), radius)

    # Scale annotation (unchanged)
    info = f"{scale.volts_per_div:.3g} V/div   {scale.secs_per_div:.3g} s/div"
    surf.blit(font_small.render(info, True, MUTED), (rect.x + 8, rect.y + 6))


def draw_time_debug_overlay(surf, rect, signal, scale, font_small):
    if signal is None or signal.time.size < 2:
        return
    t = np.asarray(signal.time, np.float64)
    y = np.asarray(signal.amplitude, np.float64)

    span = 10.0 * max(scale.secs_per_div, 1e-12)
    t0, t1 = float(scale.t_start), float(scale.t_start) + span
    t0 = max(t0, float(t[0])); t1 = min(t1, float(t[-1]))

    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    vis_n = max(0, i1 - i0)

    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if dt.size else float("nan")
    txt = [
        f"N={len(t)} vis={vis_n}",
        f"t: [{t[0]:.6g}, {t[-1]:.6g}] span={t[-1]-t[0]:.6g}",
        f"win: [{t0:.6g}, {t1:.6g}] ({span:.6g})",
        f"dt_med={dt_med:.6g}  Vpp={np.nanmax(y)-np.nanmin(y):.3g}"
    ]
    y0 = rect.y + 6
    for line in txt:
        surf.blit(font_small.render(line, True, MUTED), (rect.x + 8, y0))
        y0 += 16




def draw_waveform(surf, rect: pygame.Rect, signal: Optional[StandardSignal], scale: ViewScale, font_small):
    draw_grid(surf, rect)
    if signal is None or signal.time.size < 2:
        return

    t = np.asarray(signal.time, dtype=np.float64)
    y = np.asarray(signal.amplitude, dtype=np.float64)

    # ----- requested window (10 divisions) -----
    span = 10.0 * max(scale.secs_per_div, 1e-12)
    t0_req = float(scale.t_start)
    t1_req = t0_req + span

    # ----- intersect with available data -----
    t0 = max(float(t[0]), t0_req)
    t1 = min(float(t[-1]), t1_req)
    if not (t1 > t0):
        # snap to tail with same span
        t1 = float(t[-1])
        t0 = max(float(t[0]), t1 - span)
        if not (t1 > t0):
            return  # degenerate data

    # ----- resample uniformly: one x column per pixel -----
    cols = max(2, rect.w)
    # +1 endpoint so we always have at least 2 points even when rect.w == 1 (paranoia)
    t_grid = np.linspace(t0, t1, cols, endpoint=True)
    # interpolation strictly inside [t[0], t[-1]] due to intersection above
    y_grid = np.interp(t_grid, t, y)

    # center & scale
    v_off = float(scale.v_offset)
    vpix_per_div = rect.h / 6.0
    px_per_v = vpix_per_div / max(scale.volts_per_div, 1e-12)

    xs = rect.x + (t_grid - t0) * (rect.w / max(t1 - t0, 1e-12))
    ys = rect.centery - (y_grid - v_off) * px_per_v

    # clamp and draw
    xs = np.clip(xs, rect.x, rect.right).astype(int)
    ys = np.clip(ys, rect.y, rect.bottom - 1).astype(int)

    if xs.size >= 2:
        pts = np.column_stack((xs, ys))
        pygame.draw.aalines(surf, WAVE, False, pts)

    # Scale annotation
    info = f"{scale.volts_per_div:.3g} V/div   {scale.secs_per_div:.3g} s/div"
    surf.blit(font_small.render(info, True, MUTED), (rect.x + 8, rect.y + 6))



def draw_spectrum(
    surf, rect: pygame.Rect, freqs: np.ndarray, mag: np.ndarray,
    f0: Optional[float], font_small, spec_scale: "SpectrumScale"
):
    draw_grid(surf, rect)
    if freqs.size == 0 or mag.size == 0:
        return

    f_max = float(freqs[-1])
    if f_max <= 0:
        return

    # Visible band
    f_left = float(spec_scale.f_start)
    f_span = 10.0 * max(spec_scale.hz_per_div, 1e-9)
    f_right = min(f_max, f_left + f_span)

    # Slice bins in [f_left, f_right]
    i0 = int(np.searchsorted(freqs, f_left, side="left"))
    i1 = int(np.searchsorted(freqs, f_right, side="right"))
    if i1 - i0 < 2:
        return
    f = freqs[i0:i1]
    m = mag[i0:i1]

    # dB normalize to 0 dB top within the visible band
    m_db = 20.0 * np.log10(np.maximum(m, 1e-12))
    m_db -= np.max(m_db)
    m_db = np.clip(m_db, -100.0, 0.0)

    # Bin to columns: preserve peaks with max-in-column
    w = rect.w
    cols = w
    # Map each bin to a column
    col = ((f - f_left) / max(f_span, 1e-12) * cols).astype(int)
    col = np.clip(col, 0, cols - 1)

    col_max = np.full(cols, -100.0, dtype=np.float64)
    np.maximum.at(col_max, col, m_db)

    xs = rect.x + (np.arange(cols) / max(cols - 1, 1)) * rect.w
    ys = rect.bottom - ((col_max + 100.0) / 100.0) * rect.h

    pts = np.column_stack((xs, ys)).astype(int)
    if len(pts) >= 2:
        pygame.draw.lines(surf, SPEC, False, pts, 2)

    # f0 marker if visible
    if f0 and f_left < f0 < f_right:
        x0 = rect.x + ((f0 - f_left) / f_span) * rect.w
        pygame.draw.line(surf, (180, 220, 120), (x0, rect.y), (x0, rect.bottom), 1)

    # Title/scale
    label = f"FFT (0 dB top, 100 dB span)   {spec_scale.hz_per_div:.3g} Hz/div"
    surf.blit(font_small.render(label, True, MUTED), (rect.x + 8, rect.y + 6))




def draw_stats(surf, x, y, metrics: AnalysisMetrics, font, font_small):
    def line(txt, col=TEXT):
        nonlocal y
        surf.blit(font.render(txt, True, col), (x, y))
        y += font.get_height() + 4

    col = TEXT
    mode_txt = f"Analysis: {metrics.detected_label} | Mode={metrics.mode.title()}"
    line(mode_txt, ACCENT if metrics.mode == "auto" else YELLOW)
    line(f"f0: {metrics.f0_hz:.2f} Hz" if metrics.f0_hz else "f0: N/A", col)
    line(f"THD: {metrics.thd_percent:.2f} %" if (metrics.thd_percent is not None) else "THD: N/A", col)
    line(f"Vpp: {metrics.vpp:.3g} V" if metrics.vpp is not None else "Vpp: N/A", col)
    # line(f"Vrms: {metrics.vrms:.3g} V" if metrics.vrms is not None else "Vrms: N/A", col)
    line(f"DC: {metrics.dc:.3g} V" if metrics.dc is not None else "DC: N/A", col)

    # Conditional metrics with N/A friendly display
    duty_txt = "Duty cycle: "
    if metrics.duty_percent is None:
        duty_txt += "N/A"
        line(duty_txt, MUTED)
    else:
        line(duty_txt + f"{metrics.duty_percent:.1f} %", col)

    skew_txt = "Skew: "
    if metrics.triangle_skew_percent is None:
        skew_txt += "N/A"
        line(skew_txt, MUTED)
    else:
        line(skew_txt + f"{metrics.triangle_skew_percent:.1f} %", col)


def draw_titled_panel(surf, rect: pygame.Rect, title: str, font):
    """Draw a rounded panel with a section title."""
    pygame.draw.rect(surf, (22, 25, 35), rect, border_radius=10)
    pygame.draw.rect(surf, (70, 75, 90), rect, width=2, border_radius=10)
    surf.blit(font.render(title, True, TEXT), (rect.x + 10, rect.y + 8))


# --------------------------------------- Block 8 : End ------------------------------------------#


# -


# -


# --------------------------------------- Block 9 : App ------------------------------------------#


class App:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Digital Oscilloscope")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 16)
        self.pending_autocal = False  # one-shot autoscale when source changes

        # State
        self.signal: Optional[StandardSignal] = None
        self.scale = ViewScale(volts_per_div=1.0, secs_per_div=0.01, v_offset=0.0)
        self.mode = "auto"  # "auto","sine","square","triangle"
        self.detected_label = "Unknown (0%)"
        self.current_file: Optional[str] = None
        self.live_on = False

        # Audio
        self.audio = AudioStream(SAMPLE_RATE, 1)

        # UI layout
        self.PAD = 16
        self.SIDEBAR_W = 280
        self._layout()

        # UI widgets
        self.buttons: List["Button"] = []
        self._build_ui()

        # Zoom and scroll
        self.spec_scale: SpectrumScale = SpectrumScale(hz_per_div=1000.0, f_start=0.0)
        self._spec_fmax = 0.0
        self.pending_spec_autocal = True

    def _layout(self):
        """Compute static rectangles for the left sidebar and right plots."""
        PAD = self.PAD
        self.sidebar = pygame.Rect(PAD, PAD, self.SIDEBAR_W, HEIGHT - 2 * PAD)

        # Panel heights (tuned to your sketch)
        load_h = 160
        select_h = 190
        cal_h = 72

        # Panels (top→bottom); leave remaining space for Statistics
        self.panel_load = pygame.Rect(self.sidebar.x, self.sidebar.y, self.sidebar.w, load_h)
        self.panel_select = pygame.Rect(self.sidebar.x, self.panel_load.bottom + PAD, self.sidebar.w, select_h)
        self.panel_cal = pygame.Rect(self.sidebar.x, self.sidebar.bottom - cal_h, self.sidebar.w, cal_h)
        self.panel_stats = pygame.Rect(self.sidebar.x, self.panel_select.bottom + PAD,
                                       self.sidebar.w, self.panel_cal.y - (self.panel_select.bottom + PAD))

        # Right side plots (split view)
        right_x = self.sidebar.right + PAD
        content_w = WIDTH - right_x - PAD
        plot_h = (HEIGHT - 2 * PAD - MID_GAP) // 2
        self.plot_time = pygame.Rect(right_x, PAD, content_w, plot_h)
        self.plot_fft = pygame.Rect(right_x, PAD + plot_h + MID_GAP, content_w, plot_h)

    # ---------- UI ----------

    def _build_ui(self):
        PAD = 12
        bw, bh = self.panel_load.w - 2 * PAD, 34

        # --- Load signal panel ---
        x = self.panel_load.x + PAD
        y = self.panel_load.y + 36  # leave space for the title

        def add_here(label, cb, toggle=False):
            nonlocal y
            btn = Button((x, y, bw, bh), label, cb, toggle=toggle)
            self.buttons.append(btn)
            y += bh + 8
            return btn

        add_here("Choose file", self.on_load_csv)
        self.btn_live = add_here("Live capture", self.on_toggle_live, toggle=True)
        self.btn_live.active = False

        # --- Signal select panel (vertical buttons) ---
        x = self.panel_select.x + PAD
        y = self.panel_select.y + 36
        bw = self.panel_select.w - 2 * PAD
        self.btn_square = Button((x, y, bw, bh), "Square", lambda b: self.set_mode("square"), toggle=True);
        y += bh + 8
        self.btn_triangle = Button((x, y, bw, bh), "Triangle", lambda b: self.set_mode("triangle"), toggle=True);
        y += bh + 8
        self.btn_sine = Button((x, y, bw, bh), "Sine", lambda b: self.set_mode("sine"), toggle=True);
        y += bh + 8
        self.btn_auto = Button((x, y, bw, bh), "Auto", lambda b: self.set_mode("auto"), toggle=True)
        self.buttons += [self.btn_square, self.btn_triangle, self.btn_sine, self.btn_auto]
        self._sync_mode_buttons()

        # --- Calibration panel ---
        x = self.panel_cal.x + PAD
        y = self.panel_cal.y + 36
        bw = self.panel_cal.w - 2 * PAD
        self.buttons.append(Button((x, y, bw, bh), "Auto calibrate", self.on_auto_cal, toggle=False))

    def set_mode(self, m: str):
        self.mode = m
        self._sync_mode_buttons()

    def _sync_mode_buttons(self):
        for b in (self.btn_auto, self.btn_sine, self.btn_square, self.btn_triangle):
            b.active = False
        {"auto": self.btn_auto, "sine": self.btn_sine,
         "square": self.btn_square, "triangle": self.btn_triangle}[self.mode].active = True



    # ----- Zoom/Pan helpers -----

    def _zoom_time(self, mouse_pos, scroll_y, shift=False):
        if not self.signal:
            return
        mx, my = mouse_pos
        r = self.plot_time
        if not r.collidepoint(mx, my):
            return

        # current window
        span = 10.0 * max(self.scale.secs_per_div, 1e-12)
        # time under cursor (anchor)
        u = (mx - r.x) / max(r.w, 1)  # 0..1
        t_anchor = float(self.scale.t_start) + u * span

        if shift:
            # pan: move left/right by 10% span per wheel "notch"
            delta = -scroll_y * 0.10 * span
            self.scale.t_start += delta
        else:
            # zoom: 12% per notch; positive y => zoom in
            factor = 0.88 ** scroll_y
            new_span = span * factor
            self.scale.secs_per_div = max(new_span / 10.0, 1e-12)
            # keep anchor fixed
            self.scale.t_start = t_anchor - u * new_span

        # keep window valid
        ensure_visible_window(self.signal, self.scale)

    def _zoom_freq(self, mouse_pos, scroll_y, shift=False):
        if self.spec_scale is None or self._spec_fmax <= 0:
            return
        mx, my = mouse_pos
        r = self.plot_fft
        if not r.collidepoint(mx, my):
            return

        span = 10.0 * max(self.spec_scale.hz_per_div, 1e-9)
        u = (mx - r.x) / max(r.w, 1)  # 0..1
        f_anchor = float(self.spec_scale.f_start) + u * span

        if shift:
            # pan frequency by 10% span per notch
            delta = -scroll_y * 0.10 * span
            self.spec_scale.f_start += delta
        else:
            # zoom frequency
            factor = 0.88 ** scroll_y
            new_span = span * factor
            self.spec_scale.hz_per_div = max(new_span / 10.0, 1e-9)
            self.spec_scale.f_start = f_anchor - u * new_span

        ensure_visible_freq_window(self.spec_scale, float(self._spec_fmax))

    def _handle_wheel(self, event):
        # Pygame wheel: event.y > 0 scroll up
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        pos = pygame.mouse.get_pos()
        # Route to the plot under the cursor
        if self.plot_time.collidepoint(*pos):
            self._zoom_time(pos, event.y, shift)
        elif self.plot_fft.collidepoint(*pos):
            self._zoom_freq(pos, event.y, shift)



    # ---------- Actions ----------
    def on_load_csv(self, _=None):
        """Open a file dialog, load a scope CSV using the robust loader, calibrate the view, report status."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Important: pass 'parent=root' so dialogs stay tied to this window
        path = filedialog.askopenfilename(
            title="Choose CSV",
            filetypes=[("CSV files", "*.csv;*.CSV"), ("All files", "*.*")],
            parent=root
        )
        root.update()
        root.destroy()
        if not path:
            return

        try:
            print(f"[CSV] Loading: {path}")  # console breadcrumb
            sig = load_scope_csv_robust(path)
            if sig is None or len(sig.time) == 0:
                messagebox.showerror("CSV Load Error", "The file was selected but no usable data was parsed.")
                return
        except Exception as e:
            # load_scope_csv_robust already shows a messagebox; this is a final safety net
            messagebox.showerror("CSV Load Error (handler)", str(e))
            return

        # Success → update state/UI
        self.signal = sig
        self.pending_spec_autocal = True
        self.current_file = os.path.basename(path)
        self.scale = auto_calibrate(self.signal, self.plot_time)

        # Align the window and guarantee it contains samples
        ensure_visible_window(self.signal, self.scale)
        self.pending_autocal = False
        self.live_on = False
        if hasattr(self, "btn_live"):
            self.btn_live.active = False

        # Prime metrics immediately so the user sees an instant update
        _ = self.analyze(self.signal)
        print(f"[CSV] Loaded {self.current_file}: {len(sig.amplitude)} samples @ {sig.sampling_rate:.3g} Hz")

    def on_toggle_live(self, btn: Button):
        if not HAVE_SD:
            messagebox.showwarning("Audio", "sounddevice not installed; live capture disabled.")
            btn.active = False
            return
        if btn.active:
            # Start live capture
            try:
                self.audio.start()
                self.live_on = True
                self.current_file = None
                self.pending_autocal = True  # autoscale on first audio frame
                self.pending_spec_autocal = True
            except Exception as e:
                messagebox.showerror("Audio Error", str(e))
                btn.active = False
        else:
            self.audio.stop()
            self.live_on = False

    def on_auto_cal(self, _=None):
        if self.signal is None:
            return
        plot_rect = pygame.Rect(16, TOP_PAD, WIDTH - 280 - 32, PLOT_TOP_H)
        # self.scale = auto_calibrate(self.signal, plot_rect)
        self.scale = auto_calibrate(self.signal, self.plot_time)

    def analyze(self, sig: StandardSignal) -> AnalysisMetrics:
        """Compute metrics + (auto)classification for the current signal."""
        # Always-available metrics
        freqs, mag = compute_fft(sig)
        f0 = estimate_fundamental(freqs, mag)
        vpp, vrms, dc = compute_basic_levels(sig.amplitude)
        thd = compute_thd(freqs, mag, f0)

        # Auto-detect (with label) unless user forced a mode
        label = self.detected_label
        effective_mode = self.mode
        if self.mode == "auto":
            lbl, conf = classify_waveform(freqs, mag, sig.amplitude)
            label = f"{lbl} ({conf})"
            effective_mode = lbl.lower() if lbl in ("Sine", "Square", "Triangle") else "auto"

        # Conditional metrics
        duty = compute_duty_cycle(sig.amplitude) if effective_mode == "square" else None
        skew = compute_triangle_skew(sig.amplitude) if effective_mode == "triangle" else None

        return AnalysisMetrics(
            f0_hz=f0,
            thd_percent=thd,
            vpp=vpp,
            vrms=vrms,
            dc=dc,
            duty_percent=duty,
            triangle_skew_percent=skew,
            mode=self.mode,
            detected_label=label
        )

    # ---------- Main loop ----------

    def run(self):
        running = True
        last_ana = 0.0
        metrics = AnalysisMetrics(None, None, None, None, None, None, None, "auto", "Unknown (0%)")
        cached_fft = (np.array([]), np.array([]))

        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for b in self.buttons:
                        b.handle(event)
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_wheel(event)
            # Stream snapshot if live
            if self.live_on:
                sig = self.audio.snapshot(0.2)
                if sig is not None:
                    # Normalize time for the rolling window
                    sig.time = np.arange(len(sig.amplitude)) / sig.sampling_rate
                    self.signal = sig

            if self.pending_autocal and self.signal is not None:
                self.scale = auto_calibrate(self.signal, self.plot_time)
                self.pending_autocal = False

            # Analysis cadence
            now = time.time()
            if self.signal and (now - last_ana) > 0.1:
                metrics = self.analyze(self.signal)
                cached_fft = compute_fft(self.signal)
                self.detected_label = metrics.detected_label
                last_ana = now

            # Spectrum viewport maintenance
            freqs_tmp, _ = cached_fft
            if freqs_tmp.size:
                self._spec_fmax = float(freqs_tmp[-1])
                if self.pending_spec_autocal:
                    self.spec_scale = auto_calibrate_spectrum(freqs_tmp)
                    ensure_visible_freq_window(self.spec_scale, self._spec_fmax)
                    self.pending_spec_autocal = False
                else:
                    ensure_visible_freq_window(self.spec_scale, self._spec_fmax)

            # For CSV sources, make sure the current window is still valid
            if self.signal and self.signal.source == "csv":
                ensure_visible_window(self.signal, self.scale)

            # --- Clear ---
            self.screen.fill(BG)

            # --- Panels & titles ---
            draw_titled_panel(self.screen, self.panel_load, "Load signal", self.font)
            draw_titled_panel(self.screen, self.panel_select, "Signal select", self.font)
            draw_titled_panel(self.screen, self.panel_stats, "Statistics", self.font)
            draw_titled_panel(self.screen, self.panel_cal, "Calibration", self.font)

            # --- Panel content: Load signal status text ---
            sx = self.panel_load.x + 12
            sy = self.panel_load.y + 36 + 2 * 34 + 10  # under the two buttons
            curr = f"Current file: {self.current_file if self.current_file else '—'}"
            stat = f"Status: {'On' if self.live_on else 'Off'}"
            self.screen.blit(self.font_small.render(curr, True, TEXT), (sx, sy));
            sy += 22
            self.screen.blit(self.font_small.render(stat, True, TEXT), (sx, sy))

            # --- Draw the buttons last so they sit above panel backgrounds ---
            for b in self.buttons:
                b.draw(self.screen, self.font_small)

            # --- Right-side plots ---
            if self.signal:
                draw_waveform(self.screen, self.plot_time, self.signal, self.scale, self.font_small)
                #draw_waveform_points(self.screen, self.plot_time, self.signal, self.scale, self.font_small, radius=2)
                #draw_time_debug_overlay(self.screen, self.plot_time, self.signal, self.scale, self.font_small)

                freqs, mag = cached_fft
                draw_spectrum(self.screen, self.plot_fft, freqs, mag, metrics.f0_hz, self.font_small, self.spec_scale)


            else:
                draw_grid(self.screen, self.plot_time)
                draw_grid(self.screen, self.plot_fft)

            # --- Statistics content (inside stats panel) ---
            stats_x = self.panel_stats.x + 12
            stats_y = self.panel_stats.y + 36
            draw_stats(self.screen, stats_x, stats_y, metrics, self.font, self.font_small)

            pygame.display.flip()

        # Cleanup
        try:
            self.audio.stop()
        finally:
            pygame.quit()
# --------------------------------------- Block 9 : End ------------------------------------------#


# -


# -


# --------------------------------------- Block 10 : Run ------------------------------------------#
if __name__ == "__main__":
    # Ensure Tkinter works well with Pygame focus on Windows
    try:
        app = App()
        app.run()
    except Exception as e:
        print("Fatal error:", e)
        pygame.quit()
        sys.exit(1)
