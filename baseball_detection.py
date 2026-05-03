import torch
import cv2
import os
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# ===== CONFIGURATION =====
MODEL_PATH = "best_faster_rcnn_baseball.pth"
INPUT_VIDEO = "data/videos/IMG_0041.mov"  # Path to your raw video
OUTPUT_DIR = "outputs"               # Folder where processed videos will go
CONFIDENCE_THRESHOLD = 0.6  
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def run_inference():
    # 1. Create Output Directory and Filename
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Logic to rename output based on input
    base_name = os.path.basename(INPUT_VIDEO)
    file_name = os.path.splitext(base_name)[0]
    output_filename = f"{file_name}_detected.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Using device: {DEVICE}")
    
    # 2. Load the model
    model = get_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Successfully loaded weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Open the video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {base_name}")
    print(f"Outputting to: {output_path}")

    # 4. Process frame-by-frame
    frame_count = 0
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)

            predictions = model(img_tensor)
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            # 5. Draw detections
            for i, score in enumerate(scores):
                if score >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = boxes[i].astype(int)

                    # Draw green bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Add clean professional text label
                    label = f"BASEBALL {int(score*100)}%"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
                    
                    break # Stop at the highest confidence box

            out.write(frame)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\nProcessing complete! Final video: {output_path}")

if __name__ == "__main__":
    run_inference()