import os
import torch
from torch.utils.data import DataLoader

from dataset_loader import (
    IAMDataset,
    CVLWordsDataset,
    RIMESDataset,
    SyntheticDataset
)

from model import char2idx, HTRModel, NUM_CLASSES

import os
print("Current working directory:", os.getcwd())
# =====================================================
# DATASET PATHS (FIXED)
# =====================================================
# =====================================================
# DATASET PATHS
# =====================================================

IAM_ROOT = "../datasets/iam"

CVL_ROOT = "../datasets/cvl-database-1-1"

RIMES_ROOT = "../datasets/Handwritten2Text Training Dataset"

SYN_IMG = "../datasets/synthetic_images"

SYN_LABEL = "../datasets/synthetic_labels.csv"


# =====================================================
# PATH CHECK
# =====================================================

def check_path(path):

    if os.path.exists(path):
        print("[OK]", path)
    else:
        print("[MISSING]", path)


print("\n========================")
print("CHECKING DATASET PATHS")
print("========================")

check_path(IAM_ROOT)
check_path(CVL_ROOT)
check_path(RIMES_ROOT)
check_path(SYN_IMG)
check_path(SYN_LABEL)


# =====================================================
# LOAD DATASETS
# =====================================================

print("\n========================")
print("LOADING DATASETS")
print("========================")

try:
    iam = IAMDataset(IAM_ROOT, char2idx)
    print("IAM loaded successfully")
except Exception as e:
    print("IAM FAILED:", e)

try:
    cvl = CVLWordsDataset(CVL_ROOT, char2idx)
    print("CVL loaded successfully")
except Exception as e:
    print("CVL FAILED:", e)

try:
    rimes = RIMESDataset(RIMES_ROOT, char2idx)
    print("RIMES loaded successfully")
except Exception as e:
    print("RIMES FAILED:", e)

try:
    syn = SyntheticDataset(SYN_IMG, SYN_LABEL, char2idx)
    print("Synthetic loaded successfully")
except Exception as e:
    print("Synthetic FAILED:", e)


print("\n========================")
print("DATASET SIZES")
print("========================")

print("IAM:", len(iam))
print("CVL:", len(cvl))
print("RIMES:", len(rimes))
print("SYNTHETIC:", len(syn))


# =====================================================
# SAMPLE TEST
# =====================================================

print("\n========================")
print("SAMPLE TEST")
print("========================")

img, label, width = iam[0]

print("IAM sample width:", width)
print("IAM label length:", len(label))


# =====================================================
# DATALOADER TEST
# =====================================================

print("\n========================")
print("DATALOADER TEST")
print("========================")

from torch.utils.data import DataLoader, ConcatDataset
from dataset_loader import collate_fn

print("\n========================")
print("DATALOADER TEST")
print("========================")

# combine datasets
dataset = ConcatDataset([iam, cvl, rimes, syn])

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

for img, label, width, label_lengths in loader:

    print("Batch images shape:", img.shape)
    print("Batch label tensor size:", label.shape)
    print("Batch widths:", width)
    print("Batch label lengths:", label_lengths)

    break

print("\nDataloader working correctly")

# =====================================================
# MODEL TEST
# =====================================================

print("\n========================")
print("MODEL FORWARD TEST")
print("========================")

model = HTRModel(NUM_CLASSES)

dummy = torch.randn(2, 1, 64, 512)

out = model(dummy)

print("Forward pass successful")
print("Output shape:", out.shape)


print("\n========================")
print("ALL DIAGNOSTICS PASSED")
print("========================")