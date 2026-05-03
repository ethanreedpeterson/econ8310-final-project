# ===== FRAME EXTRACTION SCRIPT =====
# This script reads video files and XML annotations,
# then extracts only the frames where a baseball is visible and moving
# plus the empty frame immediately before and after the pitch.
# It saves everything into a master JSON file.

import os
import xml.etree.ElementTree as ET
import cv2
import json

# ===== FOLDER PATHS =====
# Assumes data is stored in a local "data/" directory
VIDEOS_DIR = "data/videos" # original video files
XML_DIR = "data/annotations" # XML labels
OUTPUT_DIR = "data/extracted_frames" # where frames will be saved

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok = True)

# ===== MAIN FUNCTION =====
def extract_frames():

    # Get all annotation files
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]

    total_saved = 0  # total frames saved across all videos
    master_labels = {} # <--- DEFINED HERE: This dictionary holds all our JSON data

    # Loop through each XML file (one per video)
    for xml_file in xml_files:

        xml_path = os.path.join(XML_DIR, xml_file)
        video_stem = os.path.splitext(xml_file)[0]

        # ===== FIND MATCHING VIDEO FILE =====
        video_path = None
        for ext in [".mov", ".mp4", ".avi", ".m4v"]:
            candidate = os.path.join(VIDEOS_DIR, video_stem + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

        # Skip if no video found
        if video_path is None:
            print(f"Skipping {xml_file}: no matching video found")
            continue

        # ===== LOAD XML ANNOTATIONS =====
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        count = 0  # number of frames saved for this video
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ===== STEP 1: GATHER ALL TARGET FRAMES & COORDINATES =====
        ball_map = {}

        for track in root.findall("track"):
            if track.attrib.get("label").lower() != "baseball":
                continue

            for box in track.findall("box"):
                # Skip frames where ball is not visible
                outside = int(box.attrib.get("outside", "0"))
                if outside == 1:
                    continue

                # Only keep frames where ball is marked as moving
                moving_attr = box.find("attribute[@name='moving']")
                if moving_attr is None or moving_attr.text.strip().lower() != "true":
                    continue

                frame_num = int(box.attrib["frame"])
                ball_map[frame_num] = [
                    float(box.attrib["xtl"]), float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]), float(box.attrib["ybr"])
                ]

        # If no moving baseballs were found, skip to the next video
        if not ball_map:
            cap.release()
            continue

        # ===== STEP 2: CALCULATE +/- 1 FRAME =====
        moving_frames = list(ball_map.keys())
        frames_to_extract = set(moving_frames)
        min_frame = min(moving_frames)
        max_frame = max(moving_frames)

        # Add the frame just before the pitch starts (if it exists)
        if min_frame - 1 >= 0:
            frames_to_extract.add(min_frame - 1)
        
        # Add the frame just after the pitch ends (if it exists)
        if max_frame + 1 < total_frames_in_video:
            frames_to_extract.add(max_frame + 1)

        # ===== STEP 3: EXTRACT AND SAVE TO JSON =====
        for frame_num in sorted(list(frames_to_extract)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()

            if not ok:
                print(f"Could not read frame {frame_num} from {video_path}")
                continue

            # ===== SAVE FRAME AS IMAGE =====
            out_name = f"{video_stem}_frame_{frame_num}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            cv2.imwrite(out_path, frame)

            # ===== SAVE COORDS TO MASTER JSON =====
            if frame_num in ball_map:
                box = ball_map[frame_num]
                # Ignore glitch boxes that are physically impossible
                if box[2] > box[0] and box[3] > box[1]:
                    master_labels[out_name] = {"label": 1, "box": box}
                else:
                    master_labels[out_name] = {"label": 0, "box": None}
            else:
                # This is one of the +/- 1 boundary frames (Empty background)
                master_labels[out_name] = {"label": 0, "box": None}

            total_saved += 1
            count += 1

            # Print progress every 50 frames
            if total_saved % 50 == 0:
                print(f"Total saved so far: {total_saved}")

        # Release video after processing
        cap.release()
        print(f"Finished {video_stem}: saved {count} frames")

    # ===== SAVE THE MASTER JSON FILE =====
    json_path = os.path.join(OUTPUT_DIR, "master_labels.json")
    with open(json_path, "w") as f:
        json.dump(master_labels, f, indent=4)

    print(f"Done. Saved {total_saved} extracted frames and master_labels.json.")

# ===== RUN SCRIPT =====
if __name__ == "__main__":
    extract_frames()