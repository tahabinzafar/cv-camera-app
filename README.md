# cv-camera-app

![Python](https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-0B23A9)
![Last commit](https://img.shields.io/github/last-commit/tahabinzafar/cv-camera-app)
![Repo size](https://img.shields.io/github/repo-size/tahabinzafar/cv-camera-app)

Two modes:

- **motion** spots movement. No model, no download, starts instantly. Good for a quick "is anything moving in my room" demo.
- **detect** names things. People, laptops, cups, dogs, whatever's in frame, using YOLO. Slower to start (it grabs a model the first time). Identifies objects.

Both draw boxes on the live feed and show your frame rate in the corner. Hit `q` to close the window.

## Getting it running

```bash
git clone https://github.com/tahabinzafar/cv-camera-app.git
cd cv-camera-app

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Heads up: that pip step pulls in PyTorch, which is big. It can sit there looking frozen for a few minutes while it unpacks. That's normal.

## Using it

Motion, the quick one:

```bash
python main.py motion
```

Object detection:

```bash
python main.py detect
```

First time you run `detect` it downloads a small model (a few MB) before the window opens, so you need to be online for that one launch. After that it's cached and works offline.

Only care about people?

```bash
python main.py detect --people-only
```

## Flags

| Flag | Mode | What it does |
|------|------|--------------|
| `--source` | both | Camera index (default `0`) or a path to a video file |
| `--min-area` | motion | Ignore moving blobs smaller than this many pixels (default 800) |
| `--save-dir` | motion | Drop a snapshot into this folder every time motion shows up |
| `--model` | detect | Which YOLO weights to load (default `yolov8n.pt`) |
| `--conf` | detect | Confidence cutoff, 0 to 1 (default 0.4) |
| `--people-only` | detect | Keep only the person class |

A few real examples:

```bash
# run on a saved video instead of the webcam
python main.py motion --source clip.mp4

# log snapshots when something moves, but ignore tiny twitches
python main.py motion --save-dir snapshots --min-area 1500

# newer, sharper model (downloads on first use)
python main.py detect --model yolo26n.pt

# bigger model, higher confidence bar
python main.py detect --model yolov8m.pt --conf 0.6
```

## Picking a model

The size letter is the whole story: `n` (nano) is fastest and least accurate, then `s`, `m`, `l`, up to `x` which is the sharpest and the slowest. On a laptop without a real GPU, start at `n`. If it's missing obvious stuff, bump up a size and keep an eye on the FPS counter. Once that drops under about 10, the feed starts to feel laggy and you've gone too far.

`yolov8n.pt` is a safe default. `yolo26n.pt` is the newer family and worth trying if you want the speed and accuracy gains. Both download themselves the first time you name them.

## How it actually works

**Motion** leans on OpenCV's MOG2 background subtractor. It learns what your static scene looks like, flags the pixels that change, cleans up the noise, and boxes anything moving that's bigger than `--min-area`. Shadows get filtered so a passing cloud doesn't set it off.

**Detect** hands each frame to a YOLO model and draws back whatever it found, with labels and confidence. The heavy lifting is all in the model; the code just feeds it frames and shows the result.

## Layout

```
cv-camera-app/
├── main.py                  # CLI, picks the mode
├── requirements.txt
├── src/
│   ├── motion_detector.py   # MOG2 motion detection
│   ├── object_detector.py   # YOLO object detection
│   └── utils.py             # FPS counter, on-screen text, camera open
└── README.md
```