# econ8310-final-project

## Overview

This project uses a **Faster R-CNN (ResNet50 + Feature Pyramid Network)** model to detect baseballs in video footage.  
The workflow converts raw videos into labeled image frames and trains a deep learning model to identify the baseball’s location in each frame.

---

## Data

The data (annotations, extracted frames, and videos) are uploaded into GitHub. Unfortunately, you must download the trained weight below, as it's file size was too large to upload to GitHub:

- **REQUIRED** [Trained Weight](https://drive.google.com/file/d/1M7Kyuet-RVBNXzw1N-hUfwEGCxORvBLw/view?usp=drive_link)

After downloading, just place the trained weight file into the root folder.

---

### Optional: Frame Extraction from Videos

The `preprocess_frames.py` script converts raw videos into individual image frames for model training. If you wish to redo this portion, feel free to do so, but it is not needed.

Run it with:

```bash
python preprocess_frames.py
```
