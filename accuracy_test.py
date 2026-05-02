
# ===== CELL 3: WRITE THE TRAINING SCRIPT TO accuracy_test.py =====
import os
import xml.etree.ElementTree as ET
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import time


############## DATASET: builds sequences of frames for CNN + LSTM
class BaseballSequenceDataset(Dataset):
    def __init__(self, frames_dir, xml_dir, image_size=(224, 224), seq_len=5):
        self.frames_dir = frames_dir
        self.xml_dir = xml_dir
        self.image_size = image_size
        self.seq_len = seq_len
        self.samples = []

        self._build_index()

    def _build_index(self):
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith(".xml")]

        for xml_file in xml_files:
            xml_path = os.path.join(self.xml_dir, xml_file)
            video_stem = os.path.splitext(xml_file)[0]

            tree = ET.parse(xml_path)
            root = tree.getroot()

            frame_boxes = {}

            for track in root.findall("track"):
                if track.attrib.get("label") != "baseball":
                    continue

                for box in track.findall("box"):
                    if int(box.attrib.get("outside", "0")) == 1:
                        continue

                    moving_attr = box.find("attribute[@name='moving']")
                    if moving_attr is None or moving_attr.text.strip().lower() != "true":
                        continue

                    frame_num = int(box.attrib["frame"])

                    frame_boxes[frame_num] = [
                        float(box.attrib["xtl"]),
                        float(box.attrib["ytl"]),
                        float(box.attrib["xbr"]),
                        float(box.attrib["ybr"])
                    ]

            sorted_frames = sorted(frame_boxes.keys())

            for i in range(len(sorted_frames) - self.seq_len + 1):
                seq_frames = sorted_frames[i:i + self.seq_len]

                if seq_frames != list(range(seq_frames[0], seq_frames[0] + self.seq_len)):
                    continue

                image_paths = [
                    os.path.join(self.frames_dir, f"{video_stem}_frame_{f}.jpg")
                    for f in seq_frames
                ]

                if not all(os.path.exists(p) for p in image_paths):
                    continue

                target_box = frame_boxes[seq_frames[-1]]

                self.samples.append({
                    "image_paths": image_paths,
                    "box": target_box
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        target_box = None

        for image_path in sample["image_paths"]:
            frame = cv2.imread(image_path)

            if frame is None:
                raise RuntimeError(f"Could not read image {image_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            orig_h, orig_w = frame.shape[:2]
            new_w, new_h = self.image_size

            ############## Resize image
            frame = cv2.resize(frame, (new_w, new_h))

            ############## Convert image to tensor
            image = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
            images.append(image)

            ############## Convert target box once
            if target_box is None:
                xtl, ytl, xbr, ybr = sample["box"]

                # Scale box to resized image
                xtl = xtl * (new_w / orig_w)
                xbr = xbr * (new_w / orig_w)
                ytl = ytl * (new_h / orig_h)
                ybr = ybr * (new_h / orig_h)

                # Convert xyxy box to center-width-height format
                cx = ((xtl + xbr) / 2) / new_w
                cy = ((ytl + ybr) / 2) / new_h
                bw = (xbr - xtl) / new_w
                bh = (ybr - ytl) / new_h

                target_box = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        images = torch.stack(images)

        return images, target_box


############## MODEL: CNN finds visual features, LSTM learns motion
class BaseballCNNLSTM(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(
            input_size=64 * 28 * 28,
            hidden_size=hidden_size,
            batch_first=True
        )

        ############## Output is cx, cy, width, height
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape

        cnn_features = []

        for t in range(seq_len):
            frame = x[:, t]
            features = self.cnn(frame)
            features = self.flatten(features)
            cnn_features.append(features)

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_out, _ = self.lstm(cnn_features)

        final_output = lstm_out[:, -1]

        return self.regressor(final_output)


############## Convert cxcywh box to xyxy box
def cxcywh_to_xyxy(box):
    cx, cy, w, h = box

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.tensor([x1, y1, x2, y2])


############## IoU evaluation metric
def compute_iou(pred, target):
    pred = cxcywh_to_xyxy(pred.detach().cpu())
    target = cxcywh_to_xyxy(target.detach().cpu())

    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    x1 = max(pred[0].item(), target[0].item())
    y1 = max(pred[1].item(), target[1].item())
    x2 = min(pred[2].item(), target[2].item())
    y2 = min(pred[3].item(), target[3].item())

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    pred_area = max(0, pred[2].item() - pred[0].item()) * max(0, pred[3].item() - pred[1].item())
    target_area = max(0, target[2].item() - target[0].item()) * max(0, target[3].item() - target[1].item())

    union_area = pred_area + target_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


############## Evaluation loop
def evaluate_model(model, loader, criterion, device, iou_threshold=0.50):
    model.eval()

    total_loss = 0.0
    all_ious = []

    with torch.no_grad():
        for images, boxes in loader:
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)

            preds = model(images)

            loss = criterion(preds, boxes)
            total_loss += loss.item()

            for pred, target in zip(preds, boxes):
                iou = compute_iou(pred, target)
                all_ious.append(iou)

    avg_loss = total_loss / len(loader)
    mean_iou = sum(all_ious) / len(all_ious)
    iou_accuracy = sum(iou >= iou_threshold for iou in all_ious) / len(all_ious)

    return avg_loss, mean_iou, iou_accuracy


############## Main script
if __name__ == "__main__":
    torch.manual_seed(34)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BaseballSequenceDataset(
        frames_dir="/content/extracted_frames",
        xml_dir="/content/annotations",
        image_size=(224, 224),
        seq_len=5
    )

    print("Total sequence samples:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError("No samples found. Check folder paths and extracted frames.")

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(34)
    )

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = BaseballCNNLSTM().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    start_time = time.time()

    ############## Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, boxes in train_loader:
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)

            optimizer.zero_grad()

            preds = model(images)
            loss = criterion(preds, boxes)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        test_loss, mean_iou, iou_accuracy = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            iou_threshold=0.50
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Test MSE: {test_loss:.4f} | "
            f"Mean IoU: {mean_iou:.4f} | "
            f"IoU >= 0.50 Accuracy: {iou_accuracy:.4f}"
        )

    ############## Final evaluation
    final_test_loss, final_mean_iou, final_iou_accuracy = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        iou_threshold=0.50
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Final Test MSE: {final_test_loss:.4f}")
    print(f"Final Mean IoU: {final_mean_iou:.4f}")
    print(f"Final IoU >= 0.50 Accuracy: {final_iou_accuracy:.4f}")

    torch.save(model.state_dict(), "/content/drive/MyDrive/baseball/cnn_lstm_cxcywh_accuracy.pth")

    print("\nSaved model to /content/drive/MyDrive/baseball/cnn_lstm_cxcywh_accuracy.pth")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
