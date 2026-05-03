# ===== FASTER R-CNN TRAINING SCRIPT =====

import os
import time
import random
import math
import numpy as np
import cv2
import torch
import torchvision
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ===== DATA AUGMENTATION =====
def augment_image_and_box(image, box):
    h, w = image.shape[:2]

    # Random brightness / contrast adjustment
    if random.random() < 0.50:
        alpha = random.uniform(0.80, 1.20) 
        beta = random.uniform(-20, 20) 
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Random slight blur
    if random.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Random small rotation
    if random.random() < 0.30:
        angle = random.uniform(-5, 5)
        cx_img, cy_img = w / 2, h / 2
        matrix = cv2.getRotationMatrix2D((cx_img, cy_img), angle, 1.0)
        
        rotated = cv2.warpAffine(
            image, matrix, (w, h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )
        image = rotated

        # Only rotate the bounding box if there actually is a baseball in this frame!
        if box is not None:
            xtl, ytl, xbr, ybr = box
            corners = np.array([[xtl, ytl], [xbr, ytl], [xbr, ybr], [xtl, ybr]], dtype=np.float32)
            ones = np.ones((corners.shape[0], 1), dtype=np.float32)
            corners_h = np.hstack([corners, ones])
            new_corners = corners_h @ matrix.T

            new_xtl = np.clip(new_corners[:, 0].min(), 0, w - 1)
            new_ytl = np.clip(new_corners[:, 1].min(), 0, h - 1)
            new_xbr = np.clip(new_corners[:, 0].max(), 0, w - 1)
            new_ybr = np.clip(new_corners[:, 1].max(), 0, h - 1)

            box = [new_xtl, new_ytl, new_xbr, new_ybr]

    return image, box


# ===== DATASET CLASS =====
class BaseballDetectionDataset(Dataset):
    def __init__(self, frames_dir, augment=False):
        self.frames_dir = frames_dir
        self.augment = augment
        self.samples = []

        # Read the single, lightning-fast JSON file instead of parsing XMLs
        json_path = os.path.join(self.frames_dir, "master_labels.json")
        with open(json_path, "r") as f:
            self.labels_dict = json.load(f)

        for img_name, data in self.labels_dict.items():
            image_path = os.path.join(self.frames_dir, img_name)
            if os.path.exists(image_path):
                self.samples.append({
                    "image_path": image_path,
                    "label": data["label"],
                    "box": data["box"]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(sample["image_path"])
        if image is None:
            raise RuntimeError(f"Could not read image: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box = sample["box"]
        label = sample["label"]

        if self.augment:
            image, box = augment_image_and_box(image, box)

        image = F.to_tensor(image)

        target = {}
        target["image_id"] = torch.tensor([idx])
        target["iscrowd"] = torch.tensor([0], dtype=torch.int64)

        # Check if the box is mathematically valid (width > 0 and height > 0)
        is_valid_box = False
        if label == 1 and box is not None:
            if (box[2] - box[0]) > 0.1 and (box[3] - box[1]) > 0.1:
                is_valid_box = True

        if is_valid_box:
            # Ball is present and box is healthy
            target["boxes"] = torch.tensor([box], dtype=torch.float32)
            target["labels"] = torch.tensor([1], dtype=torch.int64)
            target["area"] = torch.tensor([(box[2]-box[0]) * (box[3]-box[1])], dtype=torch.float32)
        else:
            # Empty background frame (or box was squashed off-screen by rotation)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        return image, target


# Needed because Faster R-CNN expects a list of dictionaries
def collate_fn(batch):
    return tuple(zip(*batch))


# Load pretrained Faster R-CNN and adjust for 1 class (baseball)
def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Measures overlap between predicted box and true box
def compute_iou(pred_box, true_box):
    x1 = max(pred_box[0], true_box[0])
    y1 = max(pred_box[1], true_box[1])
    x2 = min(pred_box[2], true_box[2])
    y2 = min(pred_box[3], true_box[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
    union_area = pred_area + true_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# Computes how well the model is detecting the baseball
def evaluate_model(model, data_loader, device, score_threshold=0.30, iou_threshold=0.50):
    model.eval()
    ious = []
    found_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Skip grading if this is an empty background frame
                if len(target["boxes"]) == 0:
                    continue

                total_images += 1
                true_box = target["boxes"][0].cpu().tolist()

                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()

                keep = scores >= score_threshold

                # If no prediction, IoU = 0
                if keep.sum() == 0:
                    ious.append(0.0)
                    continue

                # Take best prediction
                pred_box = boxes[keep][torch.argmax(scores[keep])].tolist()

                found_predictions += 1
                ious.append(compute_iou(pred_box, true_box))

    if len(ious) == 0: return 0.0, 0.0, 0.0

    mean_iou = sum(ious) / len(ious)
    iou_accuracy = sum(i >= iou_threshold for i in ious) / len(ious)
    detection_rate = found_predictions / total_images

    return mean_iou, iou_accuracy, detection_rate


# ===== MAIN SCRIPT =====
if __name__ == "__main__":
    torch.manual_seed(34)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the dataset (No XML directory needed anymore!)
    full_dataset = BaseballDetectionDataset(
        frames_dir="data/extracted_frames",
        augment=False
    )

    print("Total samples:", len(full_dataset))

    if len(full_dataset) == 0:
        raise RuntimeError("No samples found. Check extracted_frames and master_labels.json.")

    # Create reproducible train/test split
    indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(34)).tolist()
    train_size = int(0.80 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Apply augmentation to training data only
    train_dataset_full = BaseballDetectionDataset(frames_dir="data/extracted_frames", augment=True)
    test_dataset_full = BaseballDetectionDataset(frames_dir="data/extracted_frames", augment=False)

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = get_model(num_classes=2).to(device)

    model.load_state_dict(torch.load("best_faster_rcnn_baseball.pth", weights_only=True))
    print("Loaded previous best model! Resuming training...")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.00001, weight_decay=0.00005)

    scaler = torch.amp.GradScaler('cuda')

    epochs = 30
    start_time = time.time()
    best_mean_iou = 0.6523
    best_model_path = "best_faster_rcnn_baseball.pth"

    # ===== TRAINING LOOP =====
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Create the progress bar!
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True, leave=False)

        for images, targets in progress_bar:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Math check to prevent crashing
            if not math.isfinite(losses.item()):
                continue

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()
            
            # Update the progress bar text with the current math grade (loss)
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Evaluate model after each epoch
        mean_iou, iou_accuracy, detection_rate = evaluate_model(model, test_loader, device)

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new BEST model (Mean IoU: {best_mean_iou:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Mean IoU: {mean_iou:.4f} | IoU >= 0.50 Acc: {iou_accuracy:.4f} | Det Rate: {detection_rate:.4f}")

    # ===== FINAL EVALUATION =====
    final_mean_iou, final_iou_accuracy, final_detection_rate = evaluate_model(model, test_loader, device)

    print("\nFinal Test Results")
    print("------------------")
    print(f"Final Mean IoU: {final_mean_iou:.4f}")
    print(f"Final IoU >= 0.50 Accuracy: {final_iou_accuracy:.4f}")
    print(f"Final Detection Rate: {final_detection_rate:.4f}")
    print(f"\nBest Mean IoU achieved: {best_mean_iou:.4f}")
    print(f"Best model saved to: {best_model_path}")

    save_path = "faster_rcnn_baseball.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")