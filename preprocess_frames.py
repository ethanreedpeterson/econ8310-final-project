import os
import xml.etree.ElementTree as ET
import cv2


VIDEOS_DIR = "videos"
XML_DIR = "annotations"
OUTPUT_DIR = "extracted_frames"

os.makedirs(OUTPUT_DIR, exist_ok = True)


def extract_frames():
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
    total_saved = 0

    for xml_file in xml_files:
        xml_path = os.path.join(XML_DIR, xml_file)
        video_stem = os.path.splitext(xml_file)[0]

        video_path = None
        for ext in [".mov", ".mp4", ".avi", ".m4v"]:
            candidate = os.path.join(VIDEOS_DIR, video_stem + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print(f"Skipping {xml_file}: no matching video found")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        count = 0  # per-video counter

        for track in root.findall("track"):
            if track.attrib.get("label") != "baseball":
                continue

            for box in track.findall("box"):
                outside = int(box.attrib.get("outside", "0"))
                if outside == 1:
                    continue

                moving_attr = box.find("attribute[@name='moving']")
                if moving_attr is None:
                    continue

                if moving_attr.text.strip().lower() != "true":
                    continue

                frame_num = int(box.attrib["frame"])

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ok, frame = cap.read()

                if not ok:
                    print(f"Could not read frame {frame_num} from {video_path}")
                    continue

                out_name = f"{video_stem}_frame_{frame_num}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                cv2.imwrite(out_path, frame)

                total_saved += 1
                count += 1

                if total_saved % 50 == 0:
                    print(f"Total saved so far: {total_saved}")

        cap.release()

        print(f"Finished {video_stem}: saved {count} frames")

    print(f"Done. Saved {total_saved} extracted frames.")


if __name__ == "__main__":
    extract_frames()