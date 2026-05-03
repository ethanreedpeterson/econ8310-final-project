import os

# ===== FOLDER PATHS =====
VIDEOS_DIR = "data/videos"
XML_DIR = "data/annotations"
OUTPUT_DIR = "data/extracted_frames"

def audit_dataset():
    # 1. Look in the Videos folder and get the base names (without .mov)
    if os.path.exists(VIDEOS_DIR):
        video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mov', '.mp4', '.avi', '.m4v'))]
        video_stems = {os.path.splitext(f)[0] for f in video_files}
    else:
        video_stems = set()
        print(f"Warning: {VIDEOS_DIR} not found.")

    # 2. Look in the Annotations folder and get the base names (without .xml)
    if os.path.exists(XML_DIR):
        xml_files = [f for f in os.listdir(XML_DIR) if f.lower().endswith('.xml')]
        xml_stems = {os.path.splitext(f)[0] for f in xml_files}
    else:
        xml_stems = set()
        print(f"Warning: {XML_DIR} not found.")

    # 3. Look in the Extracted Frames folder and figure out which videos they came from
    extracted_stems = set()
    if os.path.exists(OUTPUT_DIR):
        frame_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.jpg')]
        for f in frame_files:
            # Our frames are named like "IMG_0030_frame_0.jpg", so we split at "_frame_"
            stem = f.split('_frame_')[0]
            extracted_stems.add(stem)

    # ===== PRINT THE AUDIT REPORT =====
    print("\n===== DATASET AUDIT REPORT =====")
    
    # Check 1: Videos that have no XML file
    missing_xml = video_stems - xml_stems
    if missing_xml:
        print(f"\n[!] Videos missing XML annotations ({len(missing_xml)}):")
        for stem in sorted(missing_xml):
            print(f"  - {stem}.mov")
            
    # Check 2: XML files that have no matching video
    missing_vid = xml_stems - video_stems
    if missing_vid:
        print(f"\n[!] XML annotations missing Video files ({len(missing_vid)}):")
        for stem in sorted(missing_vid):
            print(f"  - {stem}.xml")

    # Check 3: Videos that have both XML and Video, but ZERO frames were extracted
    valid_pairs = video_stems.intersection(xml_stems)
    zero_frames = valid_pairs - extracted_stems
    
    if zero_frames:
        print(f"\n[!] Paired videos with ZERO frames extracted ({len(zero_frames)}):")
        for stem in sorted(zero_frames):
            print(f"  - {stem}")
    else:
        print("\n[+] Success: All paired videos have at least one extracted frame!")

if __name__ == "__main__":
    audit_dataset()