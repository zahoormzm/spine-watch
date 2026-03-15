# Spine Watch

Real-time posture monitoring for the Raspberry Pi 5 using a Pi Camera V2 and MediaPipe Pose. The system measures the ear-shoulder-hip angle on each frame and alerts you when sustained slouching is detected. All processing runs in RAM — no frames are written to disk.

---

## Hardware Requirements

- Raspberry Pi 5
- Pi Camera V2 (CSI interface)
- HDMI display
- Optional: passive or active buzzer wired to GPIO 18 (BCM) and GND

## Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.10 or later
- See [Installation](#installation) for Python dependencies

---

## Installation

```bash
chmod +x setup.sh && ./setup.sh
```

The script installs system packages, creates a virtual environment at `~/postureguard-venv`, and installs all Python dependencies.

---

## Usage

Activate the environment and run:

```bash
source ~/postureguard-venv/bin/activate
python3 posture_tracker.py
```

Or run directly without activating:

```bash
~/postureguard-venv/bin/python3 posture_tracker.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `+` / `-` | Raise or lower the slouch threshold by 5 degrees |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `140` | Ear-shoulder-hip angle below which slouching is flagged (degrees) |
| `--width` | `640` | Capture width in pixels |
| `--height` | `480` | Capture height in pixels |
| `--model` | `1` | MediaPipe complexity: `0` = lite, `1` = full, `2` = heavy |
| `--no-flip` | — | Disable horizontal mirror mode |
| `--buzzer` | — | Enable GPIO buzzer alerts |
| `--buzzer-pin` | `18` | BCM pin number for the buzzer |

---

## Alert Escalation

Alerts (speaker beep + GPIO buzzer) escalate the longer you sustain bad posture without correcting it. The pattern resets as soon as posture improves.

| Continuous slouch | Cooldown | Beeps | Pitch |
|-------------------|----------|-------|-------|
| 0 – 15 s | 5.0 s | 1 | 700 Hz |
| 15 – 30 s | 3.5 s | 1 | 880 Hz |
| 30 – 60 s | 2.5 s | 2 | 1000 Hz |
| 60 – 120 s | 1.5 s | 3 | 1100 Hz |
| 120 s+ | 0.8 s | 4 | 1300 Hz |

---

## Session Summary

On exit, a summary is printed to the terminal:

```
╔════════════════════════════════════════╗
║  PostureGuard — Session Summary        ║
╠════════════════════════════════════════╣
║  Duration:              00:14:32       ║
║  Good posture:          87.3%          ║
║  Slouch episodes:       3              ║
║  Average angle:         141.5°         ║
╚════════════════════════════════════════╝
```

---

## Demo

### Video Demonstration

<!-- Replace with actual demo video -->
_Demo video coming soon._

### Terminal Output

<!-- Replace with terminal screenshot -->
_Terminal screenshot coming soon._

### Live Window

<!-- Replace with window screenshot -->
_Window screenshot coming soon._

---

## License

MIT
