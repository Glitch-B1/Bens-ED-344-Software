
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
import tkinter as tk
from tkinter import filedialog, messagebox
import re


# Try to import sounddevice; if unavailable, the app still runs (CSV mode only)
try:
    import sounddevice as sd
    HAVE_SD = True
    print("Starting audio printout\n\n")
    print(sd.get_portaudio_version())
    print(sd.query_devices())
    print("\n\nEnd of auido printout \n\n")

except Exception:
    HAVE_SD = False




# Tk for file dialog (hidden root window)


pri_once = True
# ------------------------------------------------ Block 1 : End ------------------------------------------------#

# --------------------------------------- Block 2 : Config & constants ------------------------------------------#

WIDTH, HEIGHT = 1200, 750
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
    #mode: str                               # "auto","sine","square","triangle"
    #detected_label: str                     # e.g., "Sine (82%)" or "Square (55%)"


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
            """
            print(f"[CSV] rows={len(rows)}  ncols={ncols}  header_names={header_names}")
            print(f"[CSV] t[0:3]={t[:3] if 't' in locals() else '—'}  t[-3:]={t[-3:] if 't' in locals() else '—'}")
            print(f"[CSV] y[0:3]={y[:3] if 'y' in locals() else '—'}  y[-3:]={y[-3:] if 'y' in locals() else '—'}")
            """
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
            """
            print(f"[CSV] rows={len(rows)}  ncols={ncols}  header_names={header_names}")
            print(f"[CSV] t[0:3]={t[:3] if 't' in locals() else '—'}  t[-3:]={t[-3:] if 't' in locals() else '—'}")
            print(f"[CSV] y[0:3]={y[:3] if 'y' in locals() else '—'}  y[-3:]={y[-3:] if 'y' in locals() else '—'}")
            """
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
        self.device = None  # selected PortAudio device index or name

    def _callback(self, indata, frames, time_info, status):
        # keep the original callback body
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
        try:
            kwargs = {}
            if self.device is not None:
                kwargs["device"] = self.device
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.fs,
                blocksize=AUDIO_BLOCK,
                callback=self._callback,  # <- IMPORTANT: matches method name above
                dtype='float32',
                **kwargs
            )
            self.stream.start()
        except Exception as e:
            self.running = False
            self.stream = None
            messagebox.showerror("Audio Error", str(e))

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

    def set_device(self, device):
        """
        Set input device (PortAudio index or name). If running, restart the stream.
        """
        self.device = device
        if self.running:
            try:
                self.stop()
                self.start()
            except Exception as e:
                messagebox.showerror("Audio Error", f"Could not switch device:\n{e}")

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



def list_audio_input_devices():
    """
    Return a list of input-capable devices as dicts:
    [{'index': int, 'name': str, 'max_input_channels': int, 'default_samplerate': float|None}, ...]
    Empty list if sounddevice isn't available or nothing was found.
    """
    if not HAVE_SD:
        return []
    try:
        devs = sd.query_devices()
    except Exception:
        return []
    out = []
    for i, d in enumerate(devs):
        try:
            if int(d.get("max_input_channels", 0)) > 0:
                out.append({
                    "index": i,
                    "name": d.get("name", f"Device {i}"),
                    "max_input_channels": int(d.get("max_input_channels", 0)),
                    "default_samplerate": float(d.get("default_samplerate", 0.0)) if d.get("default_samplerate") else None
                })
        except Exception:
            continue
    return out


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
def compute_triangle_skew(y: np.ndarray, smooth_win: int = 1, prominence_frac: float = 0.02) -> Optional[float]:
    y = np.asarray(y, dtype=float)
    y = y[:-1]
    mean = np.mean(y)
    #print(y)
    n = y.size
    if n < 8:
        print("No data")
        return None

    top = np.max(y)
    bottom = np.min(y)

    buf = 0.4

    # -3.4188 -0.7832 @ 0.05
    # -3.7576 -0.6864


    tip = mean + abs(mean - top)*(1-buf)
    tail = mean - abs(mean - bottom)*(1-buf)

    r_count = 0
    f_count = 0

    r_is_counting = True
    f_is_counting = False
    skews = []
    num_cycles = 1
    calc_skew = True
    #print(y)
    for dot in y:
        #print("\n\n\nTop: ", top,"\nTip: ", tip, "\nBottom: ", bottom,  "\ntail: ", tail,"\nDot = ", dot)
        if (dot < tail):
            #print("In dot < tail")
            r_is_counting = True
            f_is_counting = False

            if calc_skew == True:
                try:
                    tmp_skew = (r_count / (r_count + f_count))*100
                except ZeroDivisionError:
                    tmp_skew = 0

                skews.append(tmp_skew)
                #skew = (skew + tmp_skew) / num_cycles
                #print("\ntmp Skew: ", tmp_skew)
                calc_skew = False
                num_cycles += 1
                r_count = 0
                f_count = 0
        elif  dot > tip:
            #print("In dot > tip")
            f_is_counting = True
            r_is_counting = False
            calc_skew = True
        elif tail < dot < tip:
            #print("Between tail and tip")
            if r_is_counting == True:
                r_count += 1

            elif f_is_counting == True:
                f_count += 1


    sk = np.array(skews[1:])
    #print(sk)
    #print( sk.mean())
    #print("End \n\n\n\n\n\n\n\n\n")
    return sk.mean()





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


def find_trigger_point(y: np.ndarray, threshold: float = 0.0) -> Optional[int]:
    """
    Find the index of the first upward (rising) zero-crossing near the center of the signal.
    Returns sample index or None.
    """
    if y.size < 4:
        return None
    y = np.asarray(y)
    # Center the data roughly
    y -= np.mean(y)
    above = y >= threshold
    crossings = np.where(np.diff(above.astype(np.int8)) == 1)[0]
    if crossings.size == 0:
        return None
    # Choose the crossing nearest the middle of the signal
    mid = len(y) // 2
    idx = crossings[np.argmin(np.abs(crossings - mid))]
    return int(idx)


def rising_zero_cross_times(y: np.ndarray, fs: float, level: float = 0.0) -> np.ndarray:
    """
    Return sub-sample times (seconds) of rising crossings through 'level'.
    """
    if y.size < 4 or fs <= 0:
        return np.array([])
    y = np.asarray(y, float)
    y = y - np.mean(y)  # center to reduce bias
    above = y >= level
    edges = np.where(np.diff(above.astype(np.int8)) == 1)[0]
    if edges.size == 0:
        return np.array([])
    # linear interpolation per edge
    y0 = y[edges]
    y1 = y[edges + 1]
    frac = np.where(y1 != y0, (level - y0) / (y1 - y0), 0.0)
    t = (edges + frac) / fs
    return t


def snap_phase(t_raw: float, t_ref: float, period: Optional[float]) -> float:
    """
    Snap 't_raw' near 't_ref' by adding/subtracting integer multiples of 'period'.
    Keeps the phase consistent across frames.
    """
    if not period or period <= 0 or not np.isfinite(period):
        return t_raw
    k = round((t_ref - t_raw) / period)
    return t_raw + k * period


def rising_zero_cross_index(y: np.ndarray, level: float = 0.0) -> Optional[float]:
    """
    Sub-sample index (can be fractional) of a rising crossing near the middle.
    Returns None if not found.
    """
    if y.size < 4:
        return None
    x = y.astype(float) - np.mean(y)
    above = x >= level
    edges = np.where(np.diff(above.astype(np.int8)) == 1)[0]
    if edges.size == 0:
        return None
    mid = x.size // 2
    i = int(edges[np.argmin(np.abs(edges - mid))])
    y0, y1 = x[i], x[i+1]
    frac = 0.0 if y1 == y0 else (level - y0) / (y1 - y0)   # 0..1
    return float(i + frac)


def center_on_trigger(y: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """
    Return (y_aligned, frac_center_idx) where y is circularly rolled so the
    chosen rising crossing sits at the center sample. Also returns the
    *fractional* index of that center (for time offset).
    """
    n = y.size
    if n < 4 or fs <= 0:
        return y, (n - 1) / 2.0
    idx = rising_zero_cross_index(y, 0.0)
    if idx is None:
        return y, (n - 1) / 2.0
    c = (n - 1) / 2.0                      # desired center index (fractional)
    shift = int(round(c - idx))            # integer roll to put crossing at center
    y2 = np.roll(y, shift)
    # fractional residual after integer roll
    frac_center = idx + shift
    return y2, float(frac_center)

def _safe_vpp(y: np.ndarray) -> Optional[float]:
    """
    Safely compute the peak-to-peak voltage of a signal array.
    Returns None if invalid or empty.
    """
    if y is None or len(y) < 2:
        return None
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return None
    v = float(np.nanmax(y) - np.nanmin(y))
    return v if np.isfinite(v) else None


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
    db_per_div: float = 10.0  # NEW: vertical dB scale (dB per division, 10 divs tall => span = 10*db_per_div)


def auto_calibrate(
    signal: StandardSignal,
    plot_rect: pygame.Rect,
    cycles_target: int = 4,
    fit_margin: float = 0.90
) -> ViewScale:
    """
    Choose scales so ~cycles_target cycles are visible horizontally and
    the vertical range fills ~fit_margin of the plot height.
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

    # Try to set span from f0
    freqs, mag = compute_fft(signal)
    f0 = estimate_fundamental(freqs, mag)
    span_from_cycles = (cycles_target / f0) if (f0 and f0 > 0) else min(total_span, 0.5)

    # Require minimum samples visible
    min_vis_samples = min(max(200, int(0.02 / max(dt_med, 1e-9))), y.size)
    span_from_samples = max(min_vis_samples * dt_med, 5 * dt_med)

    visible_span = max(span_from_cycles, span_from_samples)
    visible_span = min(visible_span, max(total_span, dt_med))

    secs_per_div = max(visible_span / (10.0 * fit_margin), 1e-9)
    t_start = max(t[0], t[-1] - visible_span)

    return ViewScale(volts_per_div=volts_per_div, secs_per_div=secs_per_div,
                     v_offset=v_offset, t_start=t_start)



def auto_calibrate_spectrum(freqs: np.ndarray, fit_margin: float = 0.98) -> SpectrumScale:
    """
    Show the full available band (0..Nyquist) across 10 divisions horizontally.
    Vertical (dB) starts at 10 dB/div for a 100 dB total span (matches your current look).
    """
    if freqs.size < 2:
        return SpectrumScale(hz_per_div=1000.0, f_start=0.0, db_per_div=10.0)
    f_max = float(freqs[-1])
    span = f_max / max(fit_margin, 1e-6)
    hz_per_div = max(span / 10.0, 1e-3)
    return SpectrumScale(hz_per_div=hz_per_div, f_start=0.0, db_per_div=10.0)

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

def refit_vertical_to_signal(signal: StandardSignal, scale: "ViewScale", fit_margin: float = 0.90) -> None:
    """
    Recompute only volts/div and v_offset to fit the current signal vertically.
    Does NOT change secs_per_div or t_start.
    """
    if signal is None or signal.amplitude is None or len(signal.amplitude) < 2:
        return
    y = np.asarray(signal.amplitude, dtype=float)
    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))
    vpp = max(y_max - y_min, 1e-12)
    scale.volts_per_div = (vpp / max(fit_margin, 1e-6)) / 6.0
    scale.v_offset = 0.5 * (y_max + y_min)


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


def _inner_plot_rect(rect: pygame.Rect, x_major: int, left_axis_index: int = 1, gap_px: int = 4) -> pygame.Rect:
    """Return a rect that starts just to the right of the bold left axis."""
    axis_x = rect.x + int(round(rect.w * left_axis_index / x_major))
    x = min(rect.right - 1, axis_x + gap_px)
    return pygame.Rect(x, rect.y, rect.right - x, rect.h)


def _eng(v: float, unit: str) -> str:
    """Tiny engineering formatter (~3 sig figs)."""
    if v == 0 or not math.isfinite(v):
        return f"0 {unit}"
    ab = abs(v)
    if ab >= 1e9:  return f"{v/1e9:.3g} G{unit}"
    if ab >= 1e6:  return f"{v/1e6:.3g} M{unit}"
    if ab >= 1e3:  return f"{v/1e3:.3g} k{unit}"
    if ab >= 1:    return f"{v:.3g} {unit}"
    if ab >= 1e-3: return f"{v*1e3:.3g} m{unit}"
    if ab >= 1e-6: return f"{v*1e6:.3g} µ{unit}"
    if ab >= 1e-9: return f"{v*1e9:.3g} n{unit}"
    return f"{v:.3g} {unit}"


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

# --- Back-compat wrapper: keep existing call sites working ---
def draw_grid(surf, rect: pygame.Rect, x_divs=8, y_divs=6):
     draw_scope_grid(surf, rect, x_major=x_divs, y_major=y_divs, minors=0)



def draw_scope_grid(
    surf: pygame.Surface,
    rect: pygame.Rect,
    x_major: int = 8,
    y_major: int = 6,
    minors: int = 0,
    zero_x: Optional[int] = None,
    zero_y: Optional[int] = None,
    left_axis_at_first_major: bool = True,
):
    """
    Clean oscilloscope grid with fewer lines and a bold left axis at the first major division.
    """
    # panel background & border
    pygame.draw.rect(surf, (22, 25, 35), rect, border_radius=6)
    pygame.draw.rect(surf, AXIS, rect, width=1, border_radius=6)

    def vline(x, col, w=1): pygame.draw.line(surf, col, (x, rect.y), (x, rect.bottom), w)
    def hline(y, col, w=1): pygame.draw.line(surf, col, (rect.x, y), (rect.right, y), w)

    # optional minor grid (kept off by default)
    if minors and minors > 1:
        xm = x_major * minors
        ym = y_major * minors
        minor_col = (42, 46, 58)
        for i in range(1, xm):
            x = rect.x + int(round(rect.w * i / xm))
            vline(x, minor_col, 1)
        for j in range(1, ym):
            y = rect.y + int(round(rect.h * j / ym))
            hline(y, minor_col, 1)

    # major grid (reduced count → clearer)
    for i in range(1, x_major):
        x = rect.x + int(round(rect.w * i / x_major))
        vline(x, GRID, 2)
    for j in range(1, y_major):
        y = rect.y + int(round(rect.h * j / y_major))
        hline(y, GRID, 2)

    # emphasized zero axes (optional)
    if zero_x is not None:
        vline(int(zero_x), (110, 115, 130), 3)
    if zero_y is not None:
        hline(int(zero_y), (110, 115, 130), 3)

    # bold left axis at first major division (like your concept)
    if left_axis_at_first_major and x_major > 1:
        x0 = rect.x + int(round(rect.w * 1 / x_major))   # first major line in from the left
        vline(x0, (150, 155, 170), 4)


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


def draw_waveform(
    surf: pygame.Surface,
    rect: pygame.Rect,
    signal: Optional[StandardSignal],
    scale: ViewScale,
    font_small: pygame.font.Font,
):
    X_MAJOR, Y_MAJOR = 8, 6
    HEADER_H  = 24                          # top header band
    X_LABEL_H = font_small.get_height() + 6 # bottom band for time labels

    # --- Split areas ----------------------------------------------------------
    header_rect = pygame.Rect(rect.x, rect.y, rect.w, min(HEADER_H, rect.h))
    body_rect   = pygame.Rect(rect.x, rect.y + header_rect.h, rect.w, rect.h - header_rect.h)
    # bottom label band inside body
    xlab_h = min(X_LABEL_H, body_rect.h // 4)
    xlab_rect = pygame.Rect(body_rect.x, body_rect.bottom - xlab_h, body_rect.w, xlab_h)
    plot_rect = pygame.Rect(body_rect.x, body_rect.y, body_rect.w, body_rect.h - xlab_h)

    # Header fill + divider
    header_col = (30, 33, 43)
    pygame.draw.rect(surf, header_col, header_rect, border_radius=6)
    pygame.draw.rect(surf, (22, 25, 35), body_rect, border_radius=6)
    pygame.draw.line(surf, AXIS, (body_rect.x, body_rect.y), (body_rect.right, body_rect.y), 1)

    # --- Grid in plot area only ----------------------------------------------
    draw_scope_grid(
        surf, plot_rect, x_major=X_MAJOR, y_major=Y_MAJOR, minors=0,
        zero_y=None, left_axis_at_first_major=True
    )
    inner = _inner_plot_rect(plot_rect, x_major=X_MAJOR, left_axis_index=1, gap_px=6)

    # --- Header text ----------------------------------------------------------
    header_txt = f"{scale.volts_per_div:.3g} V/div   {scale.secs_per_div:.3g} s/div"
    surf.set_clip(header_rect)
    surf.blit(font_small.render(header_txt, True, MUTED), (header_rect.x + 8, header_rect.y + 4))
    surf.set_clip(None)

    # --- Y-axis labels (inside plot_rect, left of inner) ----------------------
    vpix_per_div = plot_rect.h / 6.0
    px_per_v = vpix_per_div / max(scale.volts_per_div, 1e-12)
    surf.set_clip(plot_rect)
    for j in range(Y_MAJOR + 1):
        y = plot_rect.y + int(round(plot_rect.h * j / Y_MAJOR))
        v_value = scale.v_offset + (plot_rect.centery - y) / px_per_v
        label = font_small.render(_eng(v_value, "V"), True, MUTED)
        lx = inner.x - label.get_width() - 6
        ly = y - label.get_height() // 2
        if ly < plot_rect.y or ly + label.get_height() > plot_rect.bottom:
            continue
        surf.blit(label, (lx, ly))
    surf.set_clip(None)

    if signal is None or signal.time.size < 2:
        # still draw empty X labels so the layout looks complete
        span = 10.0 * max(scale.secs_per_div, 1e-12)
        t0 = float(scale.t_start)
        t1 = t0 + span
    else:
        # ----- Visible window & resample into inner width ---------------------
        t = np.asarray(signal.time, dtype=np.float64)
        y = np.asarray(signal.amplitude, dtype=np.float64)
        span = 10.0 * max(scale.secs_per_div, 1e-12)
        t0 = float(scale.t_start); t1 = t0 + span
        t0 = max(float(t[0]), t0); t1 = min(float(t[-1]), t1)
        if not (t1 > t0):
            t1 = float(t[-1]); t0 = max(float(t[0]), t1 - span)
            if not (t1 > t0):
                # draw labels band and return
                pass
        cols = max(2, inner.w)
        t_grid = np.linspace(t0, t1, cols, endpoint=True)
        y_grid = np.interp(t_grid, t, y)

        xs = inner.x + (t_grid - t0) * (inner.w / max(t1 - t0, 1e-12))
        ys = plot_rect.centery - (y_grid - float(scale.v_offset)) * px_per_v
        xs = np.clip(xs, inner.x, inner.right).astype(int)
        ys = np.clip(ys, plot_rect.y, plot_rect.bottom - 1).astype(int)

        if xs.size >= 2:
            pts = np.column_stack((xs, ys))
            surf.set_clip(inner)
            pygame.draw.aalines(surf, WAVE, False, pts)
            surf.set_clip(None)

    # --- Bottom X (time) labels in dedicated band ----------------------------
    surf.set_clip(xlab_rect)
    pygame.draw.rect(surf, (22, 25, 35), xlab_rect)  # ensure clean background
    step_t = span / X_MAJOR
    for i in range(1, X_MAJOR + 1):
        t_here = t0 + i * step_t
        if t_here > t1 + 1e-15:
            break
        x = inner.x + int(round((t_here - t0) / span * inner.w))
        label = font_small.render(_eng(t_here, "s"), True, MUTED)
        x = max(inner.x, min(inner.right - label.get_width(), x - label.get_width() // 2))
        y = xlab_rect.bottom - label.get_height() - 2
        surf.blit(label, (x, y))
    surf.set_clip(None)


def draw_spectrum(
    surf: pygame.Surface,
    rect: pygame.Rect,
    freqs: np.ndarray,
    mag: np.ndarray,
    f0: Optional[float],
    font_small: pygame.font.Font,
    spec_scale: "SpectrumScale",
):
    X_MAJOR, Y_MAJOR = 8, 6
    HEADER_H  = 24
    X_LABEL_H = font_small.get_height() + 6

    # --- Split areas ----------------------------------------------------------
    header_rect = pygame.Rect(rect.x, rect.y, rect.w, min(HEADER_H, rect.h))
    body_rect   = pygame.Rect(rect.x, rect.y + header_rect.h, rect.w, rect.h - header_rect.h)
    xlab_h = min(X_LABEL_H, body_rect.h // 4)
    xlab_rect = pygame.Rect(body_rect.x, body_rect.bottom - xlab_h, body_rect.w, xlab_h)
    plot_rect = pygame.Rect(body_rect.x, body_rect.y, body_rect.w, body_rect.h - xlab_h)

    header_col = (30, 33, 43)
    pygame.draw.rect(surf, header_col, header_rect, border_radius=6)
    pygame.draw.rect(surf, (22, 25, 35), body_rect, border_radius=6)
    pygame.draw.line(surf, AXIS, (body_rect.x, body_rect.y), (body_rect.right, body_rect.y), 1)

    # Grid only in plot area
    draw_scope_grid(surf, plot_rect, x_major=X_MAJOR, y_major=Y_MAJOR, minors=0, left_axis_at_first_major=True)
    inner = _inner_plot_rect(plot_rect, x_major=X_MAJOR, left_axis_index=1, gap_px=6)

    # Header
    span_db = 10.0 * max(spec_scale.db_per_div, 1e-6)
    header = f"FFT (0 dB top, {int(span_db)} dB span)   {spec_scale.hz_per_div:.3g} Hz/div"
    surf.set_clip(header_rect)
    surf.blit(font_small.render(header, True, MUTED), (header_rect.x + 8, header_rect.y + 4))
    surf.set_clip(None)

    # Y labels (dB) inside plot_rect
    surf.set_clip(plot_rect)
    for j in range(Y_MAJOR + 1):
        y = plot_rect.y + int(round(plot_rect.h * j / Y_MAJOR))
        val_db = 0.0 - j * (span_db / Y_MAJOR)
        txt = f"{val_db:.0f} dB" if abs(val_db) >= 0.5 else "0 dB"
        srf = font_small.render(txt, True, MUTED)
        lx = inner.x - srf.get_width() - 6
        ly = y - srf.get_height() // 2
        if ly < plot_rect.y or ly + srf.get_height() > plot_rect.bottom:
            continue
        surf.blit(srf, (lx, ly))
    surf.set_clip(None)

    # Return early if no data — still draw bottom axis band
    if freqs.size == 0 or mag.size == 0:
        f_left = float(spec_scale.f_start)
        f_span = 10.0 * max(spec_scale.hz_per_div, 1e-9)
        f_right = f_left + f_span
    else:
        f_max = float(freqs[-1])
        if f_max <= 0:
            return
        f_left = float(spec_scale.f_start)
        f_span = 10.0 * max(spec_scale.hz_per_div, 1e-9)
        f_right = min(f_max, f_left + f_span)

        i0 = int(np.searchsorted(freqs, f_left, side="left"))
        i1 = int(np.searchsorted(freqs, f_right, side="right"))
        if i1 - i0 >= 2:
            f = freqs[i0:i1]
            m = mag[i0:i1]

            m_db = 20.0 * np.log10(np.maximum(m, 1e-12))
            m_db -= np.max(m_db)
            m_db = np.clip(m_db, -span_db, 0.0)

            cols = max(1, inner.w)
            col = ((f - f_left) / max(f_span, 1e-12) * cols).astype(int)
            col = np.clip(col, 0, cols - 1)
            col_max = np.full(cols, -span_db, dtype=np.float64)
            np.maximum.at(col_max, col, m_db)

            xs = inner.x + (np.arange(cols) / max(cols - 1, 1)) * inner.w
            ys = inner.bottom - ((col_max + span_db) / span_db) * inner.h

            pts = np.column_stack((xs, ys)).astype(int)
            if len(pts) >= 2:
                surf.set_clip(inner)
                pygame.draw.lines(surf, SPEC, False, pts, 2)
                surf.set_clip(None)

            if f0 and f_left < f0 < f_right:
                x0 = inner.x + ((f0 - f_left) / f_span) * inner.w
                pygame.draw.line(surf, (180, 220, 120), (x0, inner.y), (x0, inner.bottom), 1)

    # --- Bottom X (Hz) labels in dedicated band -------------------------------
    surf.set_clip(xlab_rect)
    pygame.draw.rect(surf, (22, 25, 35), xlab_rect)  # clean band
    step_f = f_span / X_MAJOR
    for i in range(1, X_MAJOR + 1):
        f_here = f_left + i * step_f
        if f_here > f_right + 1e-9:
            break
        x = inner.x + int(round((f_here - f_left) / f_span * inner.w))
        srf = font_small.render(_eng(f_here, "Hz"), True, MUTED)
        x = max(inner.x, min(inner.right - srf.get_width(), x - srf.get_width() // 2))
        y = xlab_rect.bottom - srf.get_height() - 2
        surf.blit(srf, (x, y))
    surf.set_clip(None)





def draw_stats(surf, x, y, metrics: AnalysisMetrics, font, font_small):
    """
    Render all stats unconditionally. Unavailable values show as 'N/A' in MUTED color.
    Uses existing palette constants: TEXT, MUTED.
    """
    def line(txt, col=TEXT):
        nonlocal y
        surf.blit(font.render(txt, True, col), (x, y))
        y += font.get_height() + 2

    def is_num(v):
        try:
            return v is not None and math.isfinite(float(v))
        except Exception:
            return False

    def render(label, value, fmt):
        if is_num(value):
            line(f"{label}: {fmt(value)}", TEXT)
        else:
            line(f"{label}: N/A", MUTED)

    # NOTE: Analysis line removed per request

    render("f0",               metrics.f0_hz,               lambda v: f"{v:.2f} Hz")
    render("THD",              metrics.thd_percent,         lambda v: f"{v:.2f} %")
    render("Vpp",              metrics.vpp,                 lambda v: f"{v:.3g} V")
    render("DC",               metrics.dc,                  lambda v: f"{v:.3g} V")
    render("Duty cycle",       metrics.duty_percent,        lambda v: f"{v:.1f} %")
    render("Skew",             metrics.triangle_skew_percent, lambda v: f"{v:.1f} %")


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
        #self.mode = "auto"  # "auto","sine","square","triangle"
        #self.detected_label = "Unknown (0%)"
        self.current_file: Optional[str] = None
        self.live_on = False

        # Audio
        self.audio = AudioStream(SAMPLE_RATE, 1)
        # Audio devices
        self.audio_devices = list_audio_input_devices() if HAVE_SD else []
        # Pick default input device (maps sounddevice default -> our list index)
        self.selected_dev_idx = None
        if HAVE_SD and self.audio_devices:
            try:
                default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            except Exception:
                default_in = None
            # map default_in (int) to index in our filtered list
            if isinstance(default_in, int):
                for i, d in enumerate(self.audio_devices):
                    if d["index"] == default_in:
                        self.selected_dev_idx = i
                        break
        # If no default matched, just use the first
        if self.selected_dev_idx is None and self.audio_devices:
            self.selected_dev_idx = 0
        # Apply to AudioStream instance
        if self.selected_dev_idx is not None:
            self.audio.set_device(self.audio_devices[self.selected_dev_idx]["index"])



        # UI layout
        self.PAD = 16
        self.SIDEBAR_W = 280
        self._layout()



        # Zoom and scroll
        self.spec_scale: SpectrumScale = SpectrumScale(hz_per_div=1000.0, f_start=0.0)
        self._spec_fmax = 0.0
        self.pending_spec_autocal = True





        # --- Trigger lock state (for live view) ---
        self._trig_t_smooth = None   # smoothed trigger time within the snapshot (sec)
        self._trig_period   = None   # estimated period (sec)
        self._trig_alpha    = 0.12   # smoothing factor (0..1) smaller = steadier


        # ---- Live input calibration (amplitude) — session only ----
        self.cal_gain = 1.0              # gain applied to live samples
        self.cal_active = False          # collecting calibration frames?
        self.cal_target_vpp = 1.0       # we expect 1 Vpp during calibration
        self.cal_ref_freq = 1000.0       # Hz reference tone
        self.cal_tol = 0.1              #  tolerance on frequency
        self.cal_collect_frames = 8      # frames to average
        self._cal_vpps = []              # Vpp samples collected this session
        self._cal_last_msg = "Idle"

        # Enough frames? compute new gain
        if len(self._cal_vpps) >= self.cal_collect_frames:
            med_vpp = float(np.median(self._cal_vpps))
            if med_vpp > 1e-12:
                # Target 1.0 Vpp; current display Vpp = med_vpp
                self.cal_gain = float(np.clip(self.cal_target_vpp / med_vpp, 0.01, 100.0))
                self._cal_last_msg = f"Done: Vpp≈{med_vpp:.3f} V → gain x{self.cal_gain:.3f}"

                # >>> NEW: re-fit vertical axis to the (now scaled) live signal
                if self.signal is not None:
                    refit_vertical_to_signal(self.signal, self.scale, fit_margin=0.92)
            else:
                self._cal_last_msg = "Failed: Vpp too small"
            # Stop calibration…
            self.cal_active = False
            self._cal_vpps.clear()
            if hasattr(self, "btn_input_cal"):
                self.btn_input_cal.active = False

        # UI widgets
        self.buttons: List["Button"] = []
        self._build_ui()

    def _layout(self):
        """Compute rectangles for the left sidebar and right plots."""
        PAD = self.PAD
        self.sidebar = pygame.Rect(PAD, PAD, self.SIDEBAR_W, HEIGHT - 2 * PAD)

        # Panel heights (tuned to fit content)
        load_h = 205     # ← includes room for Choose file, Live capture, and device selector
        scale_h = 150
        stats_h = 200
        PANEL_GAP = 10

        # Panels (top→bottom)
        self.panel_load = pygame.Rect(self.sidebar.x, self.sidebar.y, self.sidebar.w, load_h)
        self.panel_scale = pygame.Rect(self.sidebar.x, self.panel_load.bottom + PANEL_GAP, self.sidebar.w, scale_h)
        self.panel_stats = pygame.Rect(self.sidebar.x, self.panel_scale.bottom + PANEL_GAP, self.sidebar.w, stats_h)

        # Calibration now fills the rest
        cal_top = self.panel_stats.bottom + PANEL_GAP
        cal_h = self.sidebar.bottom - cal_top
        self.panel_cal = pygame.Rect(self.sidebar.x, cal_top, self.sidebar.w, cal_h)

        # Right-side plots (split view)
        right_x = self.sidebar.right + PAD
        content_w = WIDTH - right_x - PAD
        plot_h = (HEIGHT - 2 * PAD - MID_GAP) // 2
        self.plot_time = pygame.Rect(right_x, PAD, content_w, plot_h)
        self.plot_fft = pygame.Rect(right_x, PAD + plot_h + MID_GAP, content_w, plot_h)


    # ---------- UI ----------

    def _build_ui(self):
        PAD = 12
        BTN_H = 30
        BTN_SP = 6
        self.buttons = []  # rebuild

        # --- Load signal ---
        x = self.panel_load.x + PAD
        y = self.panel_load.y + 36
        bw = self.panel_load.w - 2 * PAD

        def add_here(label, cb, toggle=False):
            nonlocal y
            btn = Button((x, y, bw, BTN_H), label, cb, toggle=toggle)
            self.buttons.append(btn)
            y += BTN_H + BTN_SP
            return btn

        add_here("Choose file", self.on_load_csv)
        self.btn_live = add_here("Live capture", self.on_toggle_live, toggle=True)
        self.btn_live.active = False

        # --- Scale (Volts/Div, Sec/Div, dB/Div) ---
        def row_controls(panel, y0, label, on_minus, on_plus, get_value):
            x = panel.x + PAD
            bw = panel.w - 2 * PAD
            # left/right thirds for - [value] +
            w_btn = 40
            w_val = bw - 2 * w_btn - 10
            # label
            lab_surface = self.font_small.render(label, True, TEXT)
            self.screen  # (no-op to hint pyg to keep font ready)
            # buttons
            minus_btn = Button((x, y0, w_btn, BTN_H), "–", lambda b: on_minus(), toggle=False)
            val_btn = Button((x + w_btn + 5, y0, w_val, BTN_H), get_value(), lambda b: None, toggle=False)
            plus_btn = Button((x + w_btn + 5 + w_val + 5, y0, w_btn, BTN_H), "+", lambda b: on_plus(), toggle=False)
            # store a small updater for the center label (we re-render each frame)
            minus_btn._value_getter = None
            val_btn._value_getter = get_value
            plus_btn._value_getter = None
            self.buttons.extend([minus_btn, val_btn, plus_btn])
            return y0 + BTN_H + BTN_SP

        y = self.panel_scale.y + 36

        def v_get(): return f"{self.scale.volts_per_div:.3g} V/div"

        def t_get(): return f"{self.scale.secs_per_div:.3g} s/div"

        def d_get(): return f"{self.spec_scale.db_per_div:.3g} dB/div"

        y = row_controls(self.panel_scale, y,
                         "Volts/Div",
                         lambda: self._nudge_vdiv(-1),
                         lambda: self._nudge_vdiv(+1),
                         v_get)
        y = row_controls(self.panel_scale, y,
                         "Sec/Div",
                         lambda: self._nudge_tdiv(-1),
                         lambda: self._nudge_tdiv(+1),
                         t_get)
        y = row_controls(self.panel_scale, y,
                         "dB/Div",
                         lambda: self._nudge_dbdiv(-1),
                         lambda: self._nudge_dbdiv(+1),
                         d_get)

        # --- Calibration ---
        x = self.panel_cal.x + PAD
        y = self.panel_cal.y + 36 + 30
        bw = self.panel_cal.w - 2 * PAD

        # Single button that does amplitude cal (live) or view autoscale (CSV)
        self.btn_cal = Button((x, y, bw, BTN_H), "Calibrate", self.on_calibrate, toggle=True)
        self.buttons.append(self.btn_cal)

        # --- Device selector row (inside Load signal panel) ---
        if HAVE_SD:
            # Absolute position in the Load panel, below the two buttons
            dev_x  = self.panel_load.x + PAD
            dev_y  = self.panel_load.y + 36 + 2 * (BTN_H + BTN_SP)  # after "Choose file" + "Live capture"
            dev_bw = self.panel_load.w - 2 * PAD

            w_btn = 40
            w_val = dev_bw - 2 * w_btn - 10

            def _prev_dev(_=None):
                if not self.audio_devices:
                    return
                self.selected_dev_idx = (self.selected_dev_idx - 1) % len(self.audio_devices)
                self._apply_selected_device()

            def _next_dev(_=None):
                if not self.audio_devices:
                    return
                self.selected_dev_idx = (self.selected_dev_idx + 1) % len(self.audio_devices)
                self._apply_selected_device()

            def _dev_label():
                if not self.audio_devices or self.selected_dev_idx is None:
                    return "No input devices"
                name = self.audio_devices[self.selected_dev_idx]["name"]
                return (name[:28] + "…") if len(name) > 30 else name

            btn_prev = Button((dev_x, dev_y, w_btn, BTN_H), "◀", lambda b: _prev_dev())
            btn_val  = Button((dev_x + w_btn + 5, dev_y, w_val, BTN_H), _dev_label(), lambda b: None)
            btn_next = Button((dev_x + w_btn + 5 + w_val + 5, dev_y, w_btn, BTN_H), "▶", lambda b: _next_dev())
            btn_val._value_getter = _dev_label
            self.buttons.extend([btn_prev, btn_val, btn_next])



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

    def _nudge_vdiv(self, step):
        # 1/2/5 progression around current value
        v = max(self.scale.volts_per_div, 1e-12)
        base = [1, 2, 5]
        # find decade & index
        import math
        exp = int(math.floor(math.log10(v)))
        mant = v / (10 ** exp)
        idx = min(range(3), key=lambda i: abs(mant - base[i]))
        idx += step
        while idx < 0:
            idx += 3;
            exp -= 1
        while idx > 2:
            idx -= 3;
            exp += 1
        self.scale.volts_per_div = base[idx] * (10 ** exp)

    def _nudge_tdiv(self, step):
        # same 1/2/5 progression
        v = max(self.scale.secs_per_div, 1e-12)
        import math
        base = [1, 2, 5]
        exp = int(math.floor(math.log10(v)))
        mant = v / (10 ** exp)
        idx = min(range(3), key=lambda i: abs(mant - base[i]))
        idx += step
        while idx < 0:
            idx += 3;
            exp -= 1
        while idx > 2:
            idx -= 3;
            exp += 1
        self.scale.secs_per_div = base[idx] * (10 ** exp)
        # keep window anchored at the tail
        ensure_visible_window(self.signal, self.scale) if self.signal else None

    def _nudge_dbdiv(self, step):
        # clamp between 2 dB/div and 20 dB/div
        cur = float(self.spec_scale.db_per_div)
        steps = [2, 3, 5, 10, 15, 20]
        i = min(range(len(steps)), key=lambda k: abs(steps[k] - cur))
        i = max(0, min(len(steps) - 1, i + step))
        self.spec_scale.db_per_div = steps[i]

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
            #print(f"[CSV] Loading: {path}")  # console breadcrumb
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
        #print(f"[CSV] Loaded {self.current_file}: {len(sig.amplitude)} samples @ {sig.sampling_rate:.3g} Hz")

    def on_toggle_live(self, btn: Button):
        if not HAVE_SD:
            messagebox.showwarning("Audio", "sounddevice not installed; live capture disabled.")
            btn.active = False
            return
        if btn.active:
            # Start live capture
            try:

                # Ensure stream uses the currently selected device
                if HAVE_SD and self.audio_devices and self.selected_dev_idx is not None:
                    self.audio.set_device(self.audio_devices[self.selected_dev_idx]["index"])

                # Session-only: start from unity gain each live session
                self.cal_gain = 1.0
                self._cal_last_msg = "Live started → gain reset (x1.000)."
                self._cal_vpps.clear()
                self.cal_active = False
                if hasattr(self, "btn_input_cal"):
                    self.btn_input_cal.active = False
                else:
                    self.audio.stop()
                    self.live_on = False
                    self._cal_last_msg = "Live off."
                    self._cal_vpps.clear()
                    self.cal_active = False
                    self.cal_gain = 1.0
                    if hasattr(self, "btn_input_cal"):
                        self.btn_input_cal.active = False

                self.audio.start()
                self.live_on = True

                if self.signal is not None:
                    refit_vertical_to_signal(self.signal, self.scale, fit_margin=0.92)

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
        duty = compute_duty_cycle(sig.amplitude)
        skew = compute_triangle_skew(sig.amplitude)


        return AnalysisMetrics(
            f0_hz=f0,
            thd_percent=thd,
            vpp=vpp,
            vrms=vrms,
            dc=dc,
            duty_percent=duty,
            triangle_skew_percent=skew,
            #mode=self.mode,
            #detected_label=label
        )

    def _apply_selected_device(self):
        if not (HAVE_SD and self.audio_devices and self.selected_dev_idx is not None):
            return
        try:
            dev_pa_index = self.audio_devices[self.selected_dev_idx]["index"]
            self.audio.set_device(dev_pa_index)
            # Session-only: reset gain when device changes
            self.cal_gain = 1.0
            self._cal_last_msg = "Device changed → gain reset (x1.000). Press Calibrate input."
            self._cal_vpps.clear()
            self.cal_active = False
            if hasattr(self, "btn_input_cal"):
                self.btn_input_cal.active = False
        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to set input device:\n{e}")

    def on_calibrate(self, btn: "Button"):
        """
        Unified Calibrate button:
          - Live ON  -> start/stop amplitude calibration (expects 1 Vpp @ 1 kHz)
                        and immediately refits the vertical axis.
          - Live OFF -> autoscale the view (no gain change).
        """
        if self.live_on:
            # --- LIVE MODE ---
            if btn.active:
                self._cal_vpps.clear()
                self.cal_active = True
                self._cal_last_msg = "Collecting… (need 1 kHz @ 1 Vpp)"
            else:
                self.cal_active = False
                self._cal_vpps.clear()
                self._cal_last_msg = "Cancelled"

            if self.signal is not None:
                refit_vertical_to_signal(self.signal, self.scale, fit_margin=0.92)
                ensure_visible_window(self.signal, self.scale)

            pygame.event.post(pygame.event.Event(pygame.USEREVENT, {"update": True}))

        else:
            # --- STATIC/CSV MODE ---
            if self.signal is None:
                self._cal_last_msg = "No data"
            else:
                # Fit both axes for the loaded signal
                self.scale = auto_calibrate(self.signal, self.plot_time)

    # ---------- Main loop ----------

    def run(self):
        running = True
        last_ana = 0.0
        metrics = AnalysisMetrics(None, None, None, None, None, None, None)
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

            if self.live_on:
                sig = self.audio.snapshot(0.25)
                if sig is not None:
                    fs = float(sig.sampling_rate)
                    y = sig.amplitude.astype(np.float64, copy=False)

                    # 1) Phase-lock/center (you already added this)
                    y_aligned, _ = center_on_trigger(y, fs)
                    n = len(y_aligned);
                    c = (n - 1) / 2.0
                    t = (np.arange(n, dtype=np.float64) - c) / fs

                    # 2) APPLY CURRENT CALIBRATION GAIN
                    y_aligned = y_aligned * float(self.cal_gain)

                    # 3) Update signal + keep window centered
                    sig.amplitude = y_aligned
                    sig.time = t
                    span = 10.0 * max(self.scale.secs_per_div, 1e-12)
                    self.scale.t_start = -span * 0.5
                    self.signal = sig

                    # 4) If the Calibrate button is active, collect frames and compute gain
                    if self.cal_active:
                        # Estimate frequency to gate around 1 kHz
                        if 'estimate_fundamental_robust' in globals():
                            f_est = estimate_fundamental_robust(sig)
                        else:
                            freqs_tmp, mag_tmp = compute_fft(sig)
                            f_est = estimate_fundamental(freqs_tmp, mag_tmp)

                        ok_freq = (f_est is not None and abs(
                            f_est - self.cal_ref_freq) <= self.cal_tol * self.cal_ref_freq)

                        vpp = _safe_vpp(y_aligned)  # includes current gain (starts at 1.0)
                        if ok_freq and vpp is not None and vpp > 1e-9:
                            self._cal_vpps.append(vpp)
                            self._cal_last_msg = f"Collecting… {len(self._cal_vpps)}/{self.cal_collect_frames} (Vpp≈{vpp:.3f} V)"
                        else:
                            self._cal_last_msg = "Waiting for 1 kHz @ 1 Vpp…"

                        # Enough good frames? finalize gain
                        if len(self._cal_vpps) >= self.cal_collect_frames:
                            med_vpp = float(np.median(self._cal_vpps))
                            if med_vpp > 1e-12:
                                # We want 1.00 Vpp to read as 1.00 V
                                self.cal_gain = float(np.clip(self.cal_target_vpp / med_vpp, 0.01, 100.0))
                                self._cal_last_msg = f"Done: Vpp≈{med_vpp:.3f} V → gain x{self.cal_gain:.3f}"

                                # Re-fit vertical axis to the (now scaled) live signal
                                refit_vertical_to_signal(self.signal, self.scale, fit_margin=0.92)
                            else:
                                self._cal_last_msg = "Failed: Vpp too small"

                            # Stop calibration and release the button
                            self.cal_active = False
                            self._cal_vpps.clear()
                            if hasattr(self, "btn_cal"):
                                self.btn_cal.active = False

            if self.pending_autocal and self.signal is not None:
                self.scale = auto_calibrate(self.signal, self.plot_time)
                self.pending_autocal = False

            # Analysis cadence
            now = time.time()
            if self.signal and (now - last_ana) > 0.1:
                metrics = self.analyze(self.signal)
                cached_fft = compute_fft(self.signal)
                #self.detected_label = metrics.detected_label
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
            draw_titled_panel(self.screen, self.panel_scale, "Scale", self.font)
            draw_titled_panel(self.screen, self.panel_stats, "Statistics", self.font)
            draw_titled_panel(self.screen, self.panel_cal, "Calibration", self.font)

            # --- Calibration helper text ---
            cal_msg = "Input 1Vpp signal at 1kHz"
            self.screen.blit(self.font_small.render(cal_msg, True, TEXT),
                             (self.panel_cal.x + 12, self.panel_cal.y + 36))

            # --- Load status text ---
            sx = self.panel_load.x + 12
            sy = self.panel_load.y + 36 + 2 * 34 + 10

            curr = f"Current file: {self.current_file if self.current_file else '—'}"
            stat = f"Status: {'On' if self.live_on else 'Off'}"

            # New: device line
            if HAVE_SD and self.audio_devices and self.selected_dev_idx is not None:
                dev_name = self.audio_devices[self.selected_dev_idx]['name']
                # limit long device names
                if len(dev_name) > 38:
                    dev_name = dev_name[:38] + "…"
                dev_line = f"Input: {dev_name}"
            else:
                dev_line = "Input: —"

            # Draw the three lines
            self.screen.blit(self.font_small.render(curr, True, TEXT), (sx, sy)); sy += 22
            self.screen.blit(self.font_small.render(stat, True, TEXT), (sx, sy)); sy += 22
            self.screen.blit(self.font_small.render(dev_line, True, TEXT), (sx, sy))


            # --- Refresh the center labels of the three scale rows (value buttons) ---
            for b in self.buttons:
                if hasattr(b, "_value_getter") and b._value_getter:
                    # re-render the value text each frame
                    b.label = b._value_getter()

            # --- Draw buttons last so they sit above panels ---
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
                # Draw empty plots using the new header/body + lighter grid
                draw_waveform(self.screen, self.plot_time, None, self.scale, self.font_small)
                draw_spectrum(self.screen, self.plot_fft, np.array([]), np.array([]),

                              None, self.font_small, self.spec_scale)

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
