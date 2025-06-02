# Subtitle Removal
A project that removes subtitles from a video using the [E2FGVI](https://github.com/hitachinsk/E2FGVI) deep learning model.

---

## Extract a Frame with `ffmpeg`

Take a screenshot at the 360th frame:

```bash
ffmpeg -i input/sample_1min.mp4 -vf "select=eq(n\,360)" -vsync vfr -q:v 2 frame.jpg
```
## Build Docker environment:
```bash
docker build -t pixa-pipeline .
```

## Launch an Interactive Docker Shell
```bash
docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -w /app \
  pixa-pipeline
```

## Run subtitle-removal pipeline
```bash
conda run -n pixa python main.py input/sample_720.mp4 input/spec.json output/result.mp4
```
