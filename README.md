# Spine Watch

Stop slouching at your desk. Spine Watch turns your Raspberry Pi 5 into a strict (but entirely private) posture coach. Using a Pi Camera and MediaPipe Pose, it watches your ear-shoulder-hip angle in real time and beeps at you if you start to shrimp. Everything runs in RAM, so your bad posture is never saved to disk.

---

## Hardware Requirements

- Raspberry Pi 5
- Pi Camera V3 Wide (CSI interface)
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

Alerts escalate the longer you stay slouched without correcting. The sequence resets as soon as posture improves.

| Continuous slouch | Cooldown | Alert |
|-------------------|----------|-------|
| 0 – 2 s | 2.5 s | Gentle chime (soft two-note) |
| 2 – 4 s | 2.0 s | Nudge (three rising notes) |
| 4 – 7 s | 1.5 s | Warning (double two-tone burst) |
| 7 – 10 s | 1.0 s | Urgent (fast descending triplets) |
| 10 s+ | 0.8 s | Alarm (rapid siren sweep) |

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

A screen recording of Spine Watch running live is available in the repository.

---

## License

MIT
