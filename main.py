
# Main.py


# ------------------------------------------------ Block 1 : Imports ------------------------------------------------#
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





