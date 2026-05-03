# econ8310-final-project

## Overview

This project uses a **Faster R-CNN (ResNet50 + Feature Pyramid Network)** model to detect baseballs in video footage.  
The workflow converts raw videos into labeled image frames and trains a deep learning model to identify the baseball’s location in each frame.

---

## File Descriptions

```text
econ8310-final-project/
├── faster_rcnn_baseball.py      # Main training script for Faster R-CNN model
├── preprocess_frames.py         # Extracts labeled frames from raw videos
├── baseball_faster_rcnn.ipynb   # Interactive notebook used for development and experimentation
├── README.md                    # Project documentation and instructions
```

---

## Data

The dataset is not stored directly in this repository because some video files, extracted frames, annotations, and model weights exceed GitHub's size limits.

Download the folders here:

- **REQUIRED** [Extracted Frames](https://drive.google.com/drive/folders/10ScFJAOeN5-ik_oXK9Dp8kVTHYXFbTCT?usp=drive_link)
- **REQUIRED** [Annotations](https://drive.google.com/drive/folders/182HfsQ6OKfHb5d8alKnpY3_7rCytZmDs?usp=drive_link)
- **REQUIRED** [Trained Weight](https://drive.google.com/file/d/1M7Kyuet-RVBNXzw1N-hUfwEGCxORvBLw/view?usp=drive_link)
- *OPTIONAL* [Videos](https://drive.google.com/drive/folders/1wPYL3HJvZJXgiG-Z6yjGY_TpNbUf008c?usp=drive_link)

After downloading, place them into a `data/` folder like this:

```text
data/
├── annotations/
├── extracted_frames/
├── trained_model_weight/
└── videos/

```

Please note: The `videos/` folder is not required for model training if `extracted_frames/` is already available. It is only needed to regenerate frames using `preprocess_frames.py`, if you want to go that route.

---

### Optional: Frame Extraction from Videos

The `preprocess_frames.py` script converts raw videos into individual image frames for model training. If you wish to redo this portion, feel free to do so, but it is not needed.

Run it with:

```bash
python preprocess_frames.py
```
