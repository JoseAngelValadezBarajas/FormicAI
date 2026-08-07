# FormicAI

FormicAI is a personal project for observing, recording, and analyzing ant colony videos over time. The long-term vision includes computer vision, tracking, data analysis, cloud services, multimodal AI, scientific literature retrieval, and IoT telemetry.

This repository currently implements **FormicAI Vision v0.1**: video to motion analysis to metrics, heatmap, and annotated video.

It does **not** detect ants yet. In this version, the system detects moving regions.

## Requirements

- Python 3.11+
- A local video file, for example `samples/ants.mp4`

## Installation

```bash
git clone <repo-url>
cd formicai

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Unix/macOS:

```bash
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Add a Sample Video

Place an `.mp4` video inside `samples/`:

```text
samples/ants.mp4
```

Large video files in `samples/` are ignored by Git.

## Run Analysis

Using the project entry point:

```bash
python -m formicai analyze samples/ants.mp4
```

Using the convenience script:

```bash
python main.py samples/ants.mp4
```

The default motion filter scales the minimum contour area to the ROI size. For a 1920x1080 full-frame video, the default is about `1037` pixels; for a 320x240 video, it is about `38` pixels.

With a region of interest:

```bash
python main.py samples/ants.mp4 --roi 100,200,900,700
```

The ROI format is:

```text
x,y,width,height
```

If no ROI is provided, the full frame is analyzed.

By default, the first `30` frames are used to warm up the background model and are recorded with activity score `0.0`. You can change this:

```bash
python main.py samples/ants.mp4 --warmup-frames 30
```

If the video is noisy, increase sensitivity filtering:

```bash
python main.py samples/ants.mp4 --background-var-threshold 64 --min-motion-area-ratio 0.0005
```

If subtle motion is being missed, lower those values.

## Outputs

By default, files are written to `output/`:

- `output/analyzed.mp4`: original video with moving regions boxed and basic frame metrics overlaid.
- `output/activity_heatmap.png`: accumulated motion heatmap for the full video.
- `output/metrics.json`: video metadata and aggregate activity metrics.

## Activity Score

The activity score is intentionally simple in v0.1:

```text
motion pixels after noise filtering and contour filtering / total ROI pixels
```

The value is clamped between `0.0` and `1.0`.

Warmup frames are processed but not counted as motion, which avoids the first frame dominating the metrics while OpenCV initializes the background model.

This metric is useful for rough activity comparison, but it is not a biological ant count or behavior classifier.

## Tests

```bash
pytest
```

The tests focus on deterministic behavior: ROI parsing/validation, activity score calculation, contour filtering, and metrics aggregation.

## Current Limitations

- Detects moving regions, not ants.
- Background subtraction can be sensitive to lighting changes, camera shake, reflections, and early video frames while the background model stabilizes.
- No object detection, tracking, behavior classification, backend, frontend, cloud, Gemini, RAG, or IoT features are included yet.

## Next Steps

- v0.2: real ant object detection.
- v0.3: tracking and trajectories.
- v0.4: behavioral metrics.
- v0.5+: backend, frontend, cloud, AI-assisted reports, research/RAG, and IoT telemetry.
