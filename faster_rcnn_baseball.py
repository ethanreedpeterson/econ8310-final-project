# Faster R-CNN baseball detection training script

# ===== IMPORTS =====
import os
import time
import random
import numpy as np
import cv2
import torch
import torchvision
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ===== DATA AUGMENTATION =====
# Slightly alters images to help model generalize better
def augment_image_and_box(image, box):

    h, w = image.shape[:2]

    # Random brightness / contrast adjustment
    if random.random() < 0.50:
        alpha = random.uniform(0.80, 1.20) # contrast
        beta = random.uniform(-20, 20) # brightness
        image = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)

    # Random slight blur
    if random.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Random small rotation (also updates bounding box)
    if random.random() < 0.30:
        angle = random.uniform(-5, 5)

        cx_img, cy_img = w / 2, h / 2
        matrix = cv2.getRotationMatrix2D((cx_img, cy_img), angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags = cv2.INTER_LINEAR,
            borderMode = cv2.BORDER_CONSTANT,
            borderValue = (0, 0, 0)
        )

        xtl, ytl, xbr, ybr = box

        # Convert box to corner points, rotates, then rebuilds the box
        corners = np.array([
            [xtl, ytl],
            [xbr, ytl],
            [xbr, ybr],
            [xtl, ybr]
        ], dtype = np.float32)

        ones = np.ones((corners.shape[0], 1), dtype = np.float32)
        corners_h = np.hstack([corners, ones])
        new_corners = corners_h @ matrix.T

        # New bounding box after rotation
        new_xtl = np.clip(new_corners[:, 0].min(), 0, w - 1)
        new_ytl = np.clip(new_corners[:, 1].min(), 0, h - 1)
        new_xbr = np.clip(new_corners[:, 0].max(), 0, w - 1)
        new_ybr = np.clip(new_corners[:, 1].max(), 0, h - 1)

        image = rotated
        box = [new_xtl, new_ytl, new_xbr, new_ybr]

    return image, box


# ===== DATASET CLASS =====
# Converts the XML + frame images into training samples
class BaseballDetectionDataset(Dataset):

    def __init__(self, frames_dir, xml_dir, augment = False):
        self.frames_dir = frames_dir
        self.xml_dir = xml_dir
        self.augment = augment
        self.samples = []

        # Build list of valid (image, box) pairs
        self._build_index()

    def _build_index(self):
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith(".xml")]

        for xml_file in xml_files:
            xml_path = os.path.join(self.xml_dir, xml_file)
            video_stem = os.path.splitext(xml_file)[0]

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Loop through tracked objects
            for track in root.findall("track"):
                if track.attrib.get("label") != "baseball":
                    continue

                for box in track.findall("box"):

                    # Skip if ball not visible
                    if int(box.attrib.get("outside", "0")) == 1:
                        continue

                    # Only keep frames where ball is moving
                    moving_attr = box.find("attribute[@name='moving']")
                    if moving_attr is None or moving_attr.text.strip().lower() != "true":
                        continue

                    frame_num = int(box.attrib["frame"])

                    # Match XML frame to image file
                    image_name = f"{video_stem}_frame_{frame_num}.jpg"
                    image_path = os.path.join(self.frames_dir, image_name)

                    if not os.path.exists(image_path):
                        continue

                    # Bounding box coordinates
                    xtl = float(box.attrib["xtl"])
                    ytl = float(box.attrib["ytl"])
                    xbr = float(box.attrib["xbr"])
                    ybr = float(box.attrib["ybr"])

                    # Skip bad boxes
                    if xbr <= xtl or ybr <= ytl:
                        continue

                    self.samples.append({
                        "image_path": image_path,
                        "box": [xtl, ytl, xbr, ybr]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image from disk
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise RuntimeError(f"Could not read image: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box = sample["box"]

        # Apply augmentation ONLY for training data
        if self.augment:
            image, box = augment_image_and_box(image, box)

        # Convert image to PyTorch tensor
        image = F.to_tensor(image)

        # Convert box to tensor format required by Faster R-CNN
        box = torch.tensor([box], dtype = torch.float32)

        target = {
            "boxes": box,
            "labels": torch.tensor([1], dtype = torch.int64), # class 1 = baseball
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([(box[0, 2]-box[0, 0]) * (box[0, 3]-box[0, 1])]),
            "iscrowd": torch.tensor([0], dtype = torch.int64)
        }

        return image, target


# collate_fn is needed because each sample has different structure
def collate_fn(batch):
    return tuple(zip(*batch))


# ===== MODEL =====
# Load pretrained Faster R-CNN and adjust for 1 class (baseball)
def get_model(num_classes = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = "DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ===== IoU FUNCTION =====
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

    return inter_area/union_area if union_area > 0 else 0.0


# ===== EVALUATION =====
# Computes how well model is detecting the baseball
def evaluate_model(model, data_loader, device, score_threshold = 0.30, iou_threshold = 0.50):

    model.eval()

    ious = []
    found_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):

                total_images += 1
                true_box = target["boxes"][0].cpu().tolist()

                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()

                keep = scores >= score_threshold

                # If no prediction, then IoU = 0
                if keep.sum() == 0:
                    ious.append(0.0)
                    continue

                # Take best prediction
                pred_box = boxes[keep][torch.argmax(scores[keep])].tolist()

                found_predictions += 1
                ious.append(compute_iou(pred_box, true_box))

    mean_iou = sum(ious)/len(ious)
    iou_accuracy = sum(i >= iou_threshold for i in ious)/len(ious)
    detection_rate = found_predictions/total_images

    return mean_iou, iou_accuracy, detection_rate


# ===== MAIN SCRIPT =====
# This section only runs when this file is executed directly
if __name__ == "__main__":

    # Set random seed so the train/test split is repeatable
    torch.manual_seed(34)

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the full dataset without augmentation first
    # This is used only to count samples and create train/test indices
    full_dataset = BaseballDetectionDataset(
        frames_dir = "data/extracted_frames",
        xml_dir = "data/annotations",
        augment = False
    )

    print("Total samples:", len(full_dataset))

    # Stop if no usable image/annotation pairs were found
    if len(full_dataset) == 0:
        raise RuntimeError("No samples found. Check extracted_frames and annotations.")

    # Create a reproducible randomized train/test split
    indices = torch.randperm(
        len(full_dataset),
        generator = torch.Generator().manual_seed(34)
    ).tolist()

    # Use 80% of samples for training and 20% for testing
    train_size = int(0.80 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Training dataset uses augmentation
    train_dataset_full = BaseballDetectionDataset(
        frames_dir = "data/extracted_frames",
        xml_dir = "data/annotations",
        augment = True
    )

    # Testing dataset does not use augmentation
    # This keeps evaluation fair and realistic
    test_dataset_full = BaseballDetectionDataset(
        frames_dir = "data/extracted_frames",
        xml_dir = "data/annotations",
        augment = False
    )

    # Apply the train/test indices to the full datasets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    # DataLoader feeds training data to the model in small batches
    train_loader = DataLoader(
        train_dataset,
        batch_size = 2,
        shuffle = True,
        num_workers = 2,
        pin_memory = True,
        collate_fn = collate_fn
    )

    # Test loader does not shuffle so evaluation is consistent
    test_loader = DataLoader(
        test_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        collate_fn = collate_fn
    )

    # Create Faster R-CNN model and move it to GPU/CPU
    model = get_model(num_classes = 2)
    model.to(device)

    # Optimizer updates model weights during training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr = 0.0001, weight_decay = 0.0005)

    # Number of full passes through the training data
    epochs = 10

    # Track training time
    start_time = time.time()

    # Track best model based on Mean IoU
    best_mean_iou = 0
    best_model_path = "best_faster_rcnn_baseball.pth"

    # ===== TRAINING LOOP =====
    for epoch in range(epochs):

        # Put model in training mode
        model.train()

        running_loss = 0.0

        # Loop through batches of training images and targets
        for images, targets in train_loader:

            # Move images to GPU/CPU
            images = [img.to(device, non_blocking=True) for img in images]

            # Move target dictionaries to GPU/CPU
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()}
                for t in targets
            ]

            # Faster R-CNN returns loss values during training
            loss_dict = model(images, targets)

            # Combine all loss parts into one value
            losses = sum(loss for loss in loss_dict.values())

            # Clear old gradients
            optimizer.zero_grad()

            # Backpropagation
            losses.backward()

            # Update model weights
            optimizer.step()

            running_loss += losses.item()

        # Average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)

        # Evaluate model after each epoch
        mean_iou, iou_accuracy, detection_rate = evaluate_model(
            model = model,
            data_loader = test_loader,
            device = device,
            score_threshold = 0.30,
            iou_threshold = 0.50
        )

        # Save the best-performing model so far
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new BEST model (Mean IoU: {best_mean_iou:.4f})")

        # Print epoch results
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Mean IoU: {mean_iou:.4f} | "
            f"IoU >= 0.50 Accuracy: {iou_accuracy:.4f} | "
            f"Detection Rate: {detection_rate:.4f}"
        )

    # ===== FINAL EVALUATION =====
    # Evaluate the final model after all epochs finish
    final_mean_iou, final_iou_accuracy, final_detection_rate = evaluate_model(
        model = model,
        data_loader = test_loader,
        device = device,
        score_threshold = 0.30,
        iou_threshold = 0.50
    )

    # Print final model results
    print("\nFinal Test Results")
    print("------------------")
    print(f"Final Mean IoU: {final_mean_iou:.4f}")
    print(f"Final IoU >= 0.50 Accuracy: {final_iou_accuracy:.4f}")
    print(f"Final Detection Rate: {final_detection_rate:.4f}")

    # Print best model information
    print(f"\nBest Mean IoU achieved: {best_mean_iou:.4f}")
    print(f"Best model saved to: {best_model_path}")

    # Save the final epoch model separately
    save_path = "faster_rcnn_baseball.pth"
    torch.save(model.state_dict(), save_path)

    print(f"\nSaved model to {save_path}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
