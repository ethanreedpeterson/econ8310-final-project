# econ8310-final-project

## Overview

This project uses a **Faster R-CNN (ResNet50 + Feature Pyramid Network)** model to detect baseballs in video footage.  
The workflow converts raw videos into labeled image frames and trains a deep learning model to identify the baseball’s location in each frame.

---

## Data

The dataset is not stored directly in this repository because video files, extracted frames, and model weights exceed GitHub size limits.

Download the required folders here:

- [Videos](https://drive.google.com/drive/folders/1wPYL3HJvZJXgiG-Z6yjGY_TpNbUf008c?usp=drive_link)
- [Extracted Frames](https://drive.google.com/drive/folders/10ScFJAOeN5-ik_oXK9Dp8kVTHYXFbTCT?usp=drive_link)
- [Annotations](https://drive.google.com/drive/folders/182HfsQ6OKfHb5d8alKnpY3_7rCytZmDs?usp=drive_link)

After downloading, place them into a `data/` folder like this:

```text
data/
├── videos/
├── extracted_frames/
└── annotations/
```

---

### Optional: Frame Extraction from Videos

The `preprocess_frames.py` script converts raw videos into individual image frames for model training. If you wish to redo this portion, feel free to do so, but it is not needed.

Run it with:

```bash
python preprocess_frames.py
```
