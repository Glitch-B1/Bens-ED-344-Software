# main.py
# Digital Oscilloscope v0
# Windows / Python / Pygame + sounddevice
# -------------------------------------------------------------------
# Features:
# - CSV one-shot load -> static display
# - Live audio input -> continuous update
# - Auto-detect waveform type with manual override buttons
# - Split view: time-domain (upper) + FFT (lower)
# - Metrics: THD (always), f0, Vpp, Vrms, DC; Duty (square), Skew (triangle)
# - Auto-calibration of time/volt scaling
#
# Controls:
# - Load Signal -> choose CSV (expects columns: time(s), amplitude(V) OR a single amplitude column)
# - Live Capture -> toggle microphone capture
# - Mode: Auto / Sine / Square / Triangle
# - Auto (Cal) -> fit to grid
#
# Notes:
# - Uses sounddevice (PortAudio). If your default input device isn't set, select one in Windows Sound settings.

import os
import sys
import threading
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

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

# ------------------------------
# Config & constants
# ------------------------------
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



# ------------------------------
# Data structures
# ------------------------------
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

# ------------------------------
# UI helpers
# ------------------------------
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

# ------------------------------
# Signal sources
# ------------------------------
def load_csv_signal(path: str) -> Optional[StandardSignal]:
    try:
        data = np.genfromtxt(path, delimiter=',', comments='#')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # Accept two common formats:
        # 1) time, amplitude
        # 2) amplitude only -> synthesize time using inferred Fs
        if data.shape[1] >= 2:
            t = data[:, 0].astype(float)
            y = data[:, 1].astype(float)
            # infer Fs
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            fs = 1.0 / np.median(dt) if len(dt) else 0.0
        else:
            y = data[:, 0].astype(float)
            # assume uniform sampling; try to infer Fs from metadata-like filename (e.g., "..._48k.csv")
            # fallback to default
            fs = SAMPLE_RATE
            t = np.arange(len(y)) / fs

        # Clean NaNs/Infs
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m]
        y = y[m]
        # Rebase time to start at 0
        if len(t) and t[0] != 0:
            t = t - t[0]

        return StandardSignal(time=t, amplitude=y, sampling_rate=float(fs), source="csv")
    except Exception as e:
        messagebox.showerror("CSV Load Error", f"Could not read CSV:\n{e}")
        return None

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

# ------------------------------
# Analysis utilities
# ------------------------------
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

def compute_basic_levels(y: np.ndarray) -> Tuple[float, float, float]:
    if len(y) == 0:
        return (None, None, None)
    vpp = float(np.nanmax(y) - np.nanmin(y))
    vrms = float(np.sqrt(np.mean(np.square(y - np.mean(y)))))
    dc = float(np.mean(y))
    return vpp, vrms, dc

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
    # sanity check: require bimodality-ish spread
    if duty < 1.0 or duty > 99.0:
        return None
    return duty

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

def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    c = np.convolve(x, np.ones(n) / n, mode='same')
    return c

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

# ------------------------------
# Calibration
# ------------------------------
@dataclass
class ViewScale:
    # Scales for drawing
    volts_per_div: float
    secs_per_div: float
    v_offset: float  # vertical center offset (volts)
    # derived at draw time: pixels per division computed from plot rect

def auto_calibrate(signal: StandardSignal, plot_rect: pygame.Rect) -> ViewScale:
    y = signal.amplitude
    if len(y) == 0:
        return ViewScale(1.0, 0.001, 0.0)
    vpp = np.nanmax(y) - np.nanmin(y)
    vpp = vpp if vpp > 1e-9 else 1.0
    # Fit ~ 6 vertical divisions
    volts_per_div = (vpp / 0.9) / 6.0
    secs_visible = max(signal.time[-1] - signal.time[0], len(y) / signal.sampling_rate)
    # Fit ~ 10 horizontal divisions
    secs_per_div = (secs_visible / 0.9) / 10.0
    v_offset = float((np.nanmax(y) + np.nanmin(y)) * 0.5)
    return ViewScale(volts_per_div=volts_per_div, secs_per_div=secs_per_div, v_offset=v_offset)

# ------------------------------
# Drawing
# ------------------------------
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

def draw_waveform(surf, rect: pygame.Rect, signal: Optional[StandardSignal], scale: ViewScale, font_small):
    draw_grid(surf, rect)
    if signal is None or len(signal.time) == 0:
        return
    t = signal.time
    y = signal.amplitude - scale.v_offset

    # Mapping to pixels
    # Vertical: volts -> pixels
    vpix_per_div = rect.h / 6.0
    px_per_v = vpix_per_div / max(scale.volts_per_div, 1e-12)
    # Horizontal: seconds -> pixels
    hpix_per_div = rect.w / 10.0
    px_per_s = hpix_per_div / max(scale.secs_per_div, 1e-12)

    # Use visible range based on last window for audio; for CSV just draw all
    t0, t1 = t[0], t[-1]
    xs = rect.x + (t - t0) * px_per_s
    ys = rect.centery - (y * px_per_v)

    # Clip to rect and draw polyline in chunks for perf
    pts = np.stack([xs, ys], axis=1)
    # Keep only points within slightly expanded rect
    mask = (pts[:, 0] >= rect.x - 2) & (pts[:, 0] <= rect.right + 2)
    pts = pts[mask]
    if len(pts) >= 2:
        pygame.draw.lines(surf, WAVE, False, pts.astype(int), 2)

    # Axes annotations
    info = f"{scale.volts_per_div:.3g} V/div   {scale.secs_per_div:.3g} s/div"
    surf.blit(font_small.render(info, True, MUTED), (rect.x + 8, rect.y + 6))

def draw_spectrum(surf, rect: pygame.Rect, freqs: np.ndarray, mag: np.ndarray, f0: Optional[float], font_small):
    draw_grid(surf, rect)
    if len(freqs) == 0 or len(mag) == 0:
        return

    # Limit to reasonable upper bound (e.g., 8 kHz for display clarity)
    max_f = freqs[-1]
    f_limit = min(8000.0, max_f)
    idx = np.searchsorted(freqs, f_limit)
    f = freqs[:idx]
    m = mag[:idx]

    # Normalize vertical to max magnitude for drawing
    m_db = 20 * np.log10(np.maximum(m, 1e-12))
    m_db = m_db - np.max(m_db)  # top at 0 dB
    # map to pixels
    xs = rect.x + (f / f_limit) * rect.w
    ys = rect.bottom - (np.clip((m_db + 80), -80, 80) / 80.0) * rect.h  # show -80..0 dB

    pts = np.stack([xs, ys], axis=1)
    if len(pts) >= 2:
        pygame.draw.lines(surf, SPEC, False, pts.astype(int), 2)

    # annotate f0
    if f0 and f0 > 0 and f0 <= f_limit:
        x0 = rect.x + (f0 / f_limit) * rect.w
        pygame.draw.line(surf, YELLOW, (x0, rect.y), (x0, rect.bottom), 1)
        surf.blit(font_small.render(f"f0≈{f0:.1f} Hz", True, YELLOW), (x0 + 6, rect.y + 6))

    surf.blit(font_small.render("FFT (0 dB top, ~80 dB span)", True, MUTED), (rect.x + 8, rect.y + 6))

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
    line(f"Vrms: {metrics.vrms:.3g} V" if metrics.vrms is not None else "Vrms: N/A", col)
    line(f"DC: {metrics.dc:.3g} V" if metrics.dc is not None else "DC: N/A", col)

    # Conditional metrics with N/A friendly display
    duty_txt = "Duty cycle: "
    if metrics.duty_percent is None:
        duty_txt += "N/A (needs square-like)"
        line(duty_txt, MUTED)
    else:
        line(duty_txt + f"{metrics.duty_percent:.1f} %", col)

    skew_txt = "Skew: "
    if metrics.triangle_skew_percent is None:
        skew_txt += "N/A (needs triangle-like)"
        line(skew_txt, MUTED)
    else:
        line(skew_txt + f"{metrics.triangle_skew_percent:.1f} %", col)

# ------------------------------
# App state and main loop
# ------------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Digital Oscilloscope")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 16)

        # State
        self.signal: Optional[StandardSignal] = None
        self.scale = ViewScale(volts_per_div=1.0, secs_per_div=0.01, v_offset=0.0)
        self.mode = "auto"  # "auto","sine","square","triangle"
        self.detected_label = "Unknown (0%)"
        self.current_file = None
        self.live_on = False

        # Audio
        self.audio = AudioStream(SAMPLE_RATE, 1)

        # UI
        self.buttons = []
        self._build_ui()

    def _build_ui(self):
        x, y, w, h = 16, 16, 130, 36

        def add(label, cb, toggle=False):
            btn = Button((x, y, w, h), label, cb, toggle=toggle)
            self.buttons.append(btn)
            nonlocal x
            x += w + 10
            return btn

        add("Load Signal", self.on_load_csv)
        live_btn = add("Live Capture", self.on_toggle_live, toggle=True)
        live_btn.active = False

        x += 20
        add("Auto (Cal)", self.on_auto_cal)

        x += 20
        self.btn_auto = add("Auto", lambda b: self.set_mode("auto"), toggle=True)
        self.btn_sine = add("Sine", lambda b: self.set_mode("sine"), toggle=True)
        self.btn_square = add("Square", lambda b: self.set_mode("square"), toggle=True)
        self.btn_triangle = add("Triangle", lambda b: self.set_mode("triangle"), toggle=True)
        self._sync_mode_buttons()

    def set_mode(self, m: str):
        self.mode = m
        self._sync_mode_buttons()

    def _sync_mode_buttons(self):
        for b in (self.btn_auto, self.btn_sine, self.btn_square, self.btn_triangle):
            b.active = False
        if self.mode == "auto":
            self.btn_auto.active = True
        elif self.mode == "sine":
            self.btn_sine.active = True
        elif self.mode == "square":
            self.btn_square.active = True
        elif self.mode == "triangle":
            self.btn_triangle.active = True

    def on_load_csv(self, _=None):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.update()
        root.destroy()
        if not path:
            return
        sig = load_csv_signal(path)
        if sig:
            self.signal = sig
            self.current_file = os.path.basename(path)
            # Calibrate to fit the whole file
            plot_rect = pygame.Rect(16, TOP_PAD, WIDTH - 280 - 32, PLOT_TOP_H)
            self.scale = auto_calibrate(self.signal, plot_rect)

    def on_toggle_live(self, btn: Button):
        if not HAVE_SD:
            messagebox.showwarning("Audio", "sounddevice not installed.")
            btn.active = False
            return
        if btn.active:
            # turn on
            try:
                self.audio.start()
                self.live_on = True
                self.current_file = None
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
        self.scale = auto_calibrate(self.signal, plot_rect)

    def analyze(self, sig: StandardSignal) -> AnalysisMetrics:
        # FFT & f0
        freqs, mag = compute_fft(sig)
        f0 = estimate_fundamental(freqs, mag)
        vpp, vrms, dc = compute_basic_levels(sig.amplitude)
        thd = compute_thd(freqs, mag, f0)

        # Auto or forced mode
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

    def run(self):
        running = True
        last_ana = 0
        metrics = AnalysisMetrics(None, None, None, None, None, None, None, "auto", "Unknown (0%)")
        cached_fft = (np.array([]), np.array([]))

        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for b in self.buttons:
                        b.handle(event)

            # Update signal from audio if live
            if self.live_on:
                sig = self.audio.snapshot(0.2)
                if sig:
                    # Normalize time to a rolling window starting at 0
                    sig.time = np.arange(len(sig.amplitude)) / sig.sampling_rate
                    self.signal = sig

            # Periodic analysis (every ~0.1s), smooth enough for UI
            if self.signal and (time.time() - last_ana) > 0.1:
                metrics = self.analyze(self.signal)
                cached_fft = compute_fft(self.signal)
                last_ana = time.time()
                self.detected_label = metrics.detected_label

            # Draw
            self.screen.fill(BG)

            # Header/status
            hdr = f"Current file: {self.current_file if self.current_file else '—'}    Live Status: {'On' if self.live_on else 'Off'}"
            self.screen.blit(self.font.render(hdr, True, TEXT), (16, 56))

            # Plot rects
            left_w = WIDTH - 280 - 32
            plot_time = pygame.Rect(16, TOP_PAD, left_w, PLOT_TOP_H)
            plot_fft = pygame.Rect(16, BOTTOM_PAD + PLOT_TOP_H + MID_GAP, left_w, PLOT_TOP_H)

            # Draw plots
            if self.signal:
                draw_waveform(self.screen, plot_time, self.signal, self.scale, self.font_small)
                freqs, mag = cached_fft
                f0 = metrics.f0_hz
                draw_spectrum(self.screen, plot_fft, freqs, mag, f0, self.font_small)
            else:
                draw_grid(self.screen, plot_time)
                draw_grid(self.screen, plot_fft)

            # Stats panel
            panel = pygame.Rect(WIDTH - 280, TOP_PAD, 264, HEIGHT - TOP_PAD - 16)
            pygame.draw.rect(self.screen, (22, 25, 35), panel, border_radius=10)
            pygame.draw.rect(self.screen, (70, 75, 90), panel, width=2, border_radius=10)
            sx, sy = panel.x + 14, panel.y + 14
            self.screen.blit(self.font.render("Statistics", True, TEXT), (sx, sy))
            sy += 32
            draw_stats(self.screen, sx, sy, metrics, self.font, self.font_small)

            pygame.display.flip()

        # Cleanup
        try:
            self.audio.stop()
        except Exception:
            pass
        pygame.quit()

# ------------------------------
# Entry
# ------------------------------
if __name__ == "__main__":
    # Ensure Tkinter works well with Pygame focus on Windows
    try:
        app = App()
        app.run()
    except Exception as e:
        print("Fatal error:", e)
        pygame.quit()
        sys.exit(1)
