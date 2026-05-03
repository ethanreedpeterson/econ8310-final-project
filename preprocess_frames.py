# ===== FRAME EXTRACTION SCRIPT =====
# This script reads video files and XML annotations,
# then extracts only the frames where a baseball is visible and moving

import os
import xml.etree.ElementTree as ET
import cv2

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

    # Loop through each XML file (one per video)
    for xml_file in xml_files:

        xml_path = os.path.join(XML_DIR, xml_file)
        video_stem = os.path.splitext(xml_file)[0]

        # ===== FIND MATCHING VIDEO FILE =====
        # Try common video file extensions
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

        # ===== LOOP THROUGH ANNOTATED OBJECTS =====
        for track in root.findall("track"):

            # Only care about baseball objects
            if track.attrib.get("label") != "baseball":
                continue

            for box in track.findall("box"):

                # Skip frames where ball is not visible
                outside = int(box.attrib.get("outside", "0"))
                if outside == 1:
                    continue

                # Only keep frames where ball is marked as moving
                moving_attr = box.find("attribute[@name='moving']")
                if moving_attr is None:
                    continue

                if moving_attr.text.strip().lower() != "true":
                    continue

                # Get frame number from annotation
                frame_num = int(box.attrib["frame"])

                # ===== EXTRACT FRAME FROM VIDEO =====
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ok, frame = cap.read()

                if not ok:
                    print(f"Could not read frame {frame_num} from {video_path}")
                    continue

                # ===== SAVE FRAME AS IMAGE =====
                out_name = f"{video_stem}_frame_{frame_num}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                cv2.imwrite(out_path, frame)

                total_saved += 1
                count += 1

                # Print progress every 50 frames
                if total_saved % 50 == 0:
                    print(f"Total saved so far: {total_saved}")

        # Release video after processing
        cap.release()

        print(f"Finished {video_stem}: saved {count} frames")

    print(f"Done. Saved {total_saved} extracted frames.")


# ===== RUN SCRIPT =====
if __name__ == "__main__":
    extract_frames()
