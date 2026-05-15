import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

IMG_HEIGHT = 64
MAX_WIDTH  = 512

# ------------------------------------------------
# COLLATE FN
# ------------------------------------------------

def collate_fn(batch):
    images        = []
    labels        = []
    widths        = []
    label_lengths = []

    for img, label, width in batch:
        images.append(img)
        labels.append(label)
        widths.append(width)
        label_lengths.append(len(label))

    images        = torch.stack(images)
    labels        = torch.cat(labels)
    widths        = torch.tensor(widths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels, widths, label_lengths


# ------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------

def preprocess_image(img):
    if img is None:
        return None, None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords = np.column_stack(np.where(img < 250))

    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1+1, x0:x1+1]

    h, w = img.shape

    if h == 0 or w == 0:
        return None, None

    new_w = int(IMG_HEIGHT * (w / h))
    new_w = min(new_w, MAX_WIDTH)

    img = cv2.resize(img, (new_w, IMG_HEIGHT))

    canvas = np.ones((IMG_HEIGHT, MAX_WIDTH), dtype=np.uint8) * 255
    canvas[:, :new_w] = img

    img = canvas.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img, new_w


# ------------------------------------------------
# TEXT ENCODING
# ------------------------------------------------

def encode_text(text, char2idx):
    text = text.strip()
    return [char2idx[c] for c in text if c in char2idx]


# ------------------------------------------------
# BASE DATASET WITH CACHE
# ------------------------------------------------

class BaseDataset(Dataset):

    def __init__(self):
        self.cache = {}

    def load_image(self, img_path):
        if img_path in self.cache:
            return self.cache[img_path]

        try:
            img = Image.open(img_path).convert('L')
            img = np.array(img)
        except:
            return None, None

        img, width = preprocess_image(img)

        if img is None:
            return None, None

        img_tensor = torch.tensor(img, dtype=torch.float32)
        self.cache[img_path] = (img_tensor, width)

        return img_tensor, width


# ------------------------------------------------
# CVL DATASET — words + lines
#
# Words: label comes from filename stem (last part)
#        e.g. 0052-1-0-0-the.tif → "the"
#
# Lines: label comes from XML annotation file.
#        XML lives in trainset/xml/<page_id>_attributes.xml
#        Line images: trainset/lines/<writer>/<page>-<line_idx>.tif
#        Line label = concatenated word texts from attrType="2"
#        regions in order, joined by spaces.
#
# Namespace for PAGE XML:
#   http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19
# ------------------------------------------------

class CVLWordsDataset(BaseDataset):

    # PAGE XML namespace
    _NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}

    def __init__(self, root, char2idx, load_lines=True):
        super().__init__()
        print("\nLoading CVL dataset (words + lines)...")

        self.samples  = []
        self.char2idx = char2idx
        root          = Path(root)

        # Handle double-nested folder structure
        if (root / root.name / "trainset").exists():
            root = root / root.name

        # ── Words ──
        words_dir   = root / "trainset" / "words"
        word_count  = 0
        if words_dir.exists():
            for writer_folder in words_dir.iterdir():
                if not writer_folder.is_dir():
                    continue
                for img_file in writer_folder.iterdir():
                    if img_file.suffix != ".tif":
                        continue
                    parts = img_file.stem.split("-")
                    if len(parts) < 5:
                        continue
                    text = parts[-1]
                    if not text:
                        continue
                    self.samples.append((str(img_file), text))
                    word_count += 1
        print(f"  CVL words loaded: {word_count}")

        # ── Lines ──
        line_count = 0
        if load_lines:
            lines_dir = root / "trainset" / "lines"
            xml_dir   = root / "trainset" / "xml"

            if lines_dir.exists() and xml_dir.exists():
                # Build line text index from all XML files
                # key: (page_id, line_idx) → text string
                line_text_index = self._build_line_index(xml_dir)

                # Match line images to their text
                for writer_folder in lines_dir.iterdir():
                    if not writer_folder.is_dir():
                        continue
                    for img_file in writer_folder.iterdir():
                        if img_file.suffix != ".tif":
                            continue
                        # Filename: <page_id>-<line_idx>.tif
                        # e.g. 0052-1-3.tif → page_id=0052-1, line_idx=3
                        stem  = img_file.stem            # e.g. "0052-1-3"
                        parts = stem.rsplit("-", 1)       # ["0052-1", "3"]
                        if len(parts) != 2:
                            continue
                        page_id  = parts[0]
                        try:
                            line_idx = int(parts[1])
                        except ValueError:
                            continue

                        text = line_text_index.get((page_id, line_idx))
                        if not text:
                            continue

                        self.samples.append((str(img_file), text))
                        line_count += 1

        print(f"  CVL lines  loaded: {line_count}")
        print(f"  CVL total  loaded: {len(self.samples)}")

    def _build_line_index(self, xml_dir):
        """
        Parse all XML files and return a dict:
            (page_id, line_idx) → "word1 word2 word3 ..."

        XML structure relevant to us:
          AttrRegion attrType="2"  → one line
            AttrRegion attrType="1" text="word"  → one word
        Lines are ordered by their appearance in the XML (top-to-bottom).
        """
        index = {}

        for xml_file in xml_dir.iterdir():
            if not xml_file.suffix == ".xml":
                continue

            # page_id is the filename without _attributes.xml suffix
            # e.g. "0052-1_attributes.xml" → page_id = "0052-1"
            page_id = xml_file.stem.replace("_attributes", "")

            try:
                # CVL XMLs declare UTF-16 in the header but are actually
                # UTF-8 on disk. Strip the <?xml ...?> declaration line
                # before parsing to avoid the encoding conflict.
                raw   = xml_file.read_bytes()
                lines = raw.split(b"\n", 1)
                if lines[0].strip().startswith(b"<?xml"):
                    raw = lines[1] if len(lines) > 1 else raw
                root  = ET.fromstring(raw)
            except Exception:
                continue

            # Find all line-level regions (attrType="2") in document order
            # We search with and without namespace to be robust
            line_regions = []

            # Try with namespace first
            for region in root.iter("{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}AttrRegion"):
                if region.get("attrType") == "2":
                    line_regions.append(region)

            # Fallback: no namespace
            if not line_regions:
                for region in root.iter("AttrRegion"):
                    if region.get("attrType") == "2":
                        line_regions.append(region)

            for line_idx, line_region in enumerate(line_regions):
                # Collect all word texts from child attrType="1" regions
                words = []

                # With namespace
                for word_region in line_region.iter(
                    "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19}AttrRegion"
                ):
                    if word_region.get("attrType") == "1":
                        text = word_region.get("text", "").strip()
                        if text:
                            words.append(text)

                # Fallback: no namespace
                if not words:
                    for word_region in line_region.iter("AttrRegion"):
                        if word_region.get("attrType") == "1":
                            text = word_region.get("text", "").strip()
                            if text:
                                words.append(text)

                if words:
                    index[(page_id, line_idx)] = " ".join(words)

        return index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img, width = self.load_image(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        label = encode_text(text, self.char2idx)
        if not label:
            return self.__getitem__((idx + 1) % len(self.samples))
        return img, torch.tensor(label, dtype=torch.long), width


# ------------------------------------------------
# RIMES DATASET
# ------------------------------------------------

class RIMESDataset(BaseDataset):

    def __init__(self, root, char2idx):
        super().__init__()
        print("\nLoading RIMES dataset...")

        self.samples  = []
        self.char2idx = char2idx

        img_dir   = os.path.join(root, "Images")
        trans_dir = os.path.join(root, "Transcriptions")

        for txt_file in os.listdir(trans_dir):
            if not txt_file.endswith(".txt"):
                continue
            txt_path = os.path.join(trans_dir, txt_file)
            try:
                with open(txt_path, "r", encoding="utf8", errors="ignore") as f:
                    text = f.read().strip()
            except:
                continue
            img_name = txt_file.replace(".txt", ".jpg")
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, text))

        print("RIMES samples loaded:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img, width = self.load_image(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        label = encode_text(text, self.char2idx)
        return img, torch.tensor(label, dtype=torch.long), width


# ------------------------------------------------
# IAM DATASET
# ------------------------------------------------

class IAMDataset(BaseDataset):

    def __init__(self, root, char2idx):
        super().__init__()
        print("\nLoading IAM dataset...")

        self.samples  = []
        self.char2idx = char2idx

        lines_file = os.path.join(root, "ascii", "lines.txt")
        lines_dir  = os.path.join(root, "lines")

        with open(lines_file, "r", encoding="utf8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 2:
                    continue
                img_id = parts[0]
                text   = " ".join(parts[1:])
                writer = img_id.split("-")[0]
                page   = "-".join(img_id.split("-")[:2])
                img_path = os.path.join(lines_dir, writer, page, img_id + ".png")
                if os.path.exists(img_path):
                    self.samples.append((img_path, text))

        print("IAM samples loaded:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img, width = self.load_image(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        label = encode_text(text, self.char2idx)
        return img, torch.tensor(label, dtype=torch.long), width


# ------------------------------------------------
# MEDICAL PRESCRIPTION DATASET
#
# JSON format (one .json per image, same stem):
#   { "ground_truth": "<s_ocr> doctor_name: Dr. Lee ... </s>" }
#
# Since images are full prescription pages with no bounding boxes,
# this loader:
#   1. Parses all field values from the ground_truth string
#   2. Detects text line regions via horizontal projection
#   3. Pairs detected lines with parsed values in order
#   4. Each (cropped_line_image, text) = one training sample
# ------------------------------------------------

import json
import re


class MedicalPrescriptionDataset(BaseDataset):

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, root, char2idx, splits=("train", "val")):
        super().__init__()
        print("\nLoading Medical Prescription dataset (splits: {})...".format(splits))

        self.samples  = []
        self.char2idx = char2idx

        for split in splits:
            img_dir = os.path.join(root, split, "images")
            ann_dir = os.path.join(root, split, "annotations")

            if not os.path.isdir(img_dir):
                print("  WARNING: {} not found".format(img_dir))
                continue
            if not os.path.isdir(ann_dir):
                print("  WARNING: {} not found".format(ann_dir))
                continue

            loaded  = 0
            skipped = 0

            for fname in sorted(os.listdir(img_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.IMG_EXTS:
                    continue

                stem     = os.path.splitext(fname)[0]
                img_path = os.path.join(img_dir, fname)
                ann_path = os.path.join(ann_dir, stem + ".json")

                if not os.path.exists(ann_path):
                    skipped += 1
                    continue

                field_values = self._parse_ground_truth(ann_path)
                if not field_values:
                    skipped += 1
                    continue

                line_imgs = self._detect_lines(img_path)
                if not line_imgs:
                    skipped += 1
                    continue

                for line_img, text in zip(line_imgs, field_values):
                    text = text.strip()
                    if not text:
                        continue
                    text_filtered = "".join(c for c in text if c in char2idx)
                    if not text_filtered:
                        continue
                    self.samples.append((line_img, text_filtered))
                    loaded += 1

            print("  [{}] loaded={} line-samples  skipped={} images".format(
                split, loaded, skipped))

        print("Medical Prescription samples loaded: {}".format(len(self.samples)))

    def _parse_ground_truth(self, ann_path):
        """
        Parse the ground_truth string into a list of text values.
        Input:  "<s_ocr> doctor_name: Dr. Lee clinic_name: X medications: - A - B </s>"
        Output: ["Dr. Lee", "X", "A", "B"]
        """
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        raw = data.get("ground_truth", "")
        # Strip XML-style tags
        raw = re.sub(r"<[^>]+>", "", raw).strip()

        # Split on "field_name:" patterns — key is word chars/underscores + colon
        parts = re.split(r"\b[\w_]+:", raw)

        values = []
        for part in parts[1:]:   # first element is empty (before first field)
            part = part.strip()
            if not part:
                continue
            # Medication lists use " - " as bullet separator
            if " - " in part:
                med_items = [m.strip().lstrip("-").strip()
                             for m in part.split(" - ") if m.strip()]
                values.extend(med_items)
            else:
                values.append(part)

        return [v for v in values if v]

    def _detect_lines(self, img_path):
        """
        Detect text line regions via horizontal projection profile.
        Returns list of preprocessed line image tensors.
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.array(Image.open(img_path).convert("L"))
        except Exception:
            return []

        if img is None or img.size == 0:
            return []

        h, w = img.shape

        # Binarise — text is dark on light background
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Horizontal projection: sum of dark pixels per row
        proj = binary.sum(axis=1).astype(np.float32)

        # Smooth to merge nearby strokes within the same line
        kernel      = np.ones(5) / 5
        proj_smooth = np.convolve(proj, kernel, mode="same")

        # Find row spans above threshold
        threshold  = proj_smooth.max() * 0.05
        in_line    = False
        line_spans = []
        start      = 0

        for row in range(h):
            if not in_line and proj_smooth[row] > threshold:
                in_line = True
                start   = row
            elif in_line and proj_smooth[row] <= threshold:
                in_line = False
                line_spans.append((start, row))
        if in_line:
            line_spans.append((start, h))

        # Remove very short spans (noise, punctuation blobs)
        line_spans = [(s, e) for s, e in line_spans if (e - s) >= 10]

        if not line_spans:
            return []

        # Merge spans separated by tiny gaps (split ascenders/descenders)
        merged = [line_spans[0]]
        for s, e in line_spans[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e < 8:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))

        # Crop, pad, and preprocess each line
        line_tensors = []
        pad = 4

        for s, e in merged:
            y0   = max(0, s - pad)
            y1   = min(h, e + pad)
            crop = img[y0:y1, :]

            processed, _ = preprocess_image(crop)
            if processed is None:
                continue

            line_tensors.append(torch.tensor(processed, dtype=torch.float32))

        return line_tensors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        line_img, text = self.samples[idx]
        label = encode_text(text, self.char2idx)
        if not label:
            return self.__getitem__((idx + 1) % len(self.samples))
        width = int(line_img.shape[2]) if line_img.dim() == 3 else MAX_WIDTH
        return line_img, torch.tensor(label, dtype=torch.long), width

# ------------------------------------------------
# BD DATASET
#
# Folder structure:
#   bd-dataset/
#     Training/
#       training_words/    <- image files (.png)
#       training_labels.csv
#     Testing/
#       training_words/
#       training_labels.csv
#     Validation/
#       training_words/
#       training_labels.csv
#
# CSV format (tab or comma separated):
#   IMAGE    MEDICINE_NAME    GENERIC_NAME
#   0.png    Aceta            Paracetamol
#
# Only MEDICINE_NAME is used as the training label.
# ------------------------------------------------

class BDDataset(BaseDataset):

    # Per-split folder and CSV names
    SPLIT_WORDS_MAP = {
        "Training":  ("training_words",   "training_labels.csv"),
        "Testing":   ("testing_words",    "testing_labels.csv"),
        "Validation":("validation_words", "validation_labels.csv"),
    }

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, root, char2idx, splits=("train", "val")):
        super().__init__()
        print("\nLoading BD dataset (splits: {})...".format(splits))

        self.samples  = []
        self.char2idx = char2idx

        for split in splits:
            # Resolve folder name
            folder_name = {
                "train": "Training", "training": "Training",
                "test":  "Testing",  "testing":  "Testing",
                "val":   "Validation", "validation": "Validation",
            }.get(split.lower(), split)

            words_folder, csv_name = self.SPLIT_WORDS_MAP.get(
                folder_name, ("training_words", "training_labels.csv")
            )

            split_dir  = os.path.join(root, folder_name)
            words_dir  = os.path.join(split_dir, words_folder)
            csv_path   = os.path.join(split_dir, csv_name)

            if not os.path.isdir(split_dir):
                print("  WARNING: split folder not found: {}".format(split_dir))
                continue
            if not os.path.isdir(words_dir):
                print("  WARNING: words folder not found: {}".format(words_dir))
                continue
            if not os.path.exists(csv_path):
                print("  WARNING: labels CSV not found: {}".format(csv_path))
                continue

            # Load CSV — try tab then comma separator
            try:
                df = pd.read_csv(csv_path, sep="\t")
                if "IMAGE" not in df.columns:
                    df = pd.read_csv(csv_path, sep=",")
            except Exception as e:
                print("  WARNING: could not read CSV {}: {}".format(csv_path, e))
                continue

            # Validate required columns
            if "IMAGE" not in df.columns or "MEDICINE_NAME" not in df.columns:
                print("  WARNING: CSV missing IMAGE or MEDICINE_NAME column: {}".format(csv_path))
                continue

            loaded  = 0
            skipped = 0

            for _, row in df.iterrows():
                img_name = str(row["IMAGE"]).strip()
                text     = str(row["MEDICINE_NAME"]).strip()

                # Skip empty or nan labels
                if not text or text.lower() == "nan":
                    skipped += 1
                    continue

                # Try exact filename first, then with common extensions
                img_path = os.path.join(words_dir, img_name)
                if not os.path.exists(img_path):
                    # Try appending .png if no extension given
                    stem = os.path.splitext(img_name)[0]
                    found = False
                    for ext in self.IMG_EXTS:
                        candidate = os.path.join(words_dir, stem + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            found    = True
                            break
                    if not found:
                        skipped += 1
                        continue

                # Filter text to known vocabulary
                text_filtered = "".join(c for c in text if c in char2idx)
                if not text_filtered:
                    skipped += 1
                    continue

                self.samples.append((img_path, text_filtered))
                loaded += 1

            print("  [{}] loaded={}  skipped={}".format(folder_name, loaded, skipped))

        print("BD dataset samples loaded: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img, width = self.load_image(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        label = encode_text(text, self.char2idx)
        if not label:
            return self.__getitem__((idx + 1) % len(self.samples))
        return img, torch.tensor(label, dtype=torch.long), width