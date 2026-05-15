import torch
from dataset_loader import CVLWordsDataset, IAMDataset, RIMESDataset, SyntheticDataset
from htr_model_v2 import char2idx

# Dataset paths
IAM_ROOT = "../datasets/iam"
CVL_ROOT = "../datasets/cvl-database-1-1"
RIMES_ROOT = "../datasets/Handwritten2Text Training Dataset"
SYN_IMG = "../datasets/synthetic_images"
SYN_LABEL = "../datasets/synthetic_labels.csv"

print("Loading datasets...")

# Load one dataset for inspection
cvl_dataset = CVLWordsDataset(CVL_ROOT, char2idx)
iam_dataset = IAMDataset(IAM_ROOT, char2idx)
rimes_dataset = RIMESDataset(RIMES_ROOT, char2idx)
syn_dataset = SyntheticDataset(SYN_IMG, SYN_LABEL, char2idx)

datasets = {
    "CVL": cvl_dataset,
    "IAM": iam_dataset,
    "RIMES": rimes_dataset,
    "Synthetic": syn_dataset
}

print("\nDataset sizes:")
for name, dataset in datasets.items():
    print(f"  {name}: {len(dataset)} samples")

print("\n" + "="*60)
print("INSPECTING SAMPLES")
print("="*60)

for name, dataset in datasets.items():
    if len(dataset) > 0:
        print(f"\n{name} Dataset Sample [0]:")
        sample = dataset[0]
        print(f"  type(sample) = {type(sample)}")
        print(f"  len(sample) = {len(sample)}")

        for i, item in enumerate(sample):
            if hasattr(item, 'shape'):
                print(f"  [{i}] type={type(item)}, shape={item.shape}")
            else:
                print(f"  [{i}] type={type(item)}, value={item}")

        # Show decoded text if it's a label
        if len(sample) >= 2:
            label = sample[1]
            if isinstance(label, torch.Tensor):
                label_ints = label.tolist()
                print(f"  Raw label indices: {label_ints}")

                # Try to decode using idx2char
                from htr_model_v2 import idx2char
                decoded_chars = []
                for idx in label_ints:
                    if idx in idx2char:
                        decoded_chars.append(idx2char[idx])
                    else:
                        decoded_chars.append(f"UNK({idx})")

                print(f"  Decoded chars: {decoded_chars}")

                # Final decoded text
                decoded_text = "".join([idx2char[idx] for idx in label_ints if idx in idx2char and idx != 0])
                print(f"  Decoded text: '{decoded_text}'")

        print("-" * 40)
        break  # Just show first dataset