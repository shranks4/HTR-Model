import torch
import os
import editdistance
from torch.utils.data import DataLoader
from dataset_loader import CVLWordsDataset, collate_fn
from htr_model_v2 import HTRModelV2, NUM_CLASSES, char2idx, idx2char, greedy_decode, MAX_DECODE_LEN

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =====================================================
# LOAD MODEL
# =====================================================

def load_model(model_path):
    model = HTRModelV2(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

# =====================================================
# TEST DATASET CLASS
# =====================================================

class CVLTestDataset(CVLWordsDataset):
    def __init__(self, root, char2idx):
        super().__init__(root, char2idx)
        print("\nLoading CVL TEST WORD dataset...")
        self.samples = []
        words_dir = os.path.join(root, "testset", "words")
        for writer_folder in os.listdir(words_dir):
            writer_path = os.path.join(words_dir, writer_folder)
            if not os.path.isdir(writer_path):
                continue
            for img_file in os.listdir(writer_path):
                if not img_file.endswith(".tif"):
                    continue
                parts = img_file.split("-")
                if len(parts) < 5:
                    continue
                text = parts[-1].split(".")[0]
                img_path = os.path.join(writer_path, img_file)
                self.samples.append((img_path, text))
        print("CVL TEST WORD samples loaded:", len(self.samples))

# =====================================================
# COMPUTE METRICS
# =====================================================

def compute_cer(pred_text, true_text):
    """Compute Character Error Rate"""
    if len(true_text) == 0:
        return 0 if len(pred_text) == 0 else 1
    return editdistance.eval(pred_text, true_text) / len(true_text)

def compute_wer(pred_text, true_text):
    """Compute Word Error Rate"""
    pred_words = pred_text.split()
    true_words = true_text.split()
    if len(true_words) == 0:
        return 0 if len(pred_words) == 0 else 1
    return editdistance.eval(pred_words, true_words) / len(true_words)

# =====================================================
# TEST FUNCTION
# =====================================================

def test_model(model, test_loader):
    model.eval()

    total_cer = 0
    total_wer = 0
    num_samples = 0

    print("\nTesting model...")

    with torch.no_grad():
        for batch_idx, (images, labels, widths, label_lengths) in enumerate(test_loader):
            images = images.to(DEVICE)

            # Forward pass
            logits1, logits2 = model(images, num_decode_steps=MAX_DECODE_LEN)

            # Use refined output (logits2)
            pred_texts = greedy_decode(logits2)

            # Decode true labels
            batch_size = images.size(0)
            true_texts = []
            start_idx = 0
            for i in range(batch_size):
                end_idx = start_idx + label_lengths[i].item()
                label_seq = labels[start_idx:end_idx]
                true_text = "".join([idx2char[idx.item()] for idx in label_seq if idx.item() in idx2char])
                true_texts.append(true_text)
                start_idx = end_idx

            # Compute metrics
            for pred_text, true_text in zip(pred_texts, true_texts):
                cer = compute_cer(pred_text, true_text)
                wer = compute_wer(pred_text, true_text)

                total_cer += cer
                total_wer += wer
                num_samples += 1

                if batch_idx < 5:  # Print first few examples
                    print(f"Pred: '{pred_text}' | True: '{true_text}' | CER: {cer:.3f}")

            if batch_idx % 20 == 0:
                print(f"Processed {batch_idx * test_loader.batch_size} samples...")

    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples

    print("\n" + "="*60)
    print("Test Results (HTRModelV2):")
    print("="*60)
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Total samples: {num_samples}")
    print("="*60)

    return avg_cer, avg_wer

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    # Load model
    model_path = "best_htr_model_v2.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        exit(1)

    model = load_model(model_path)
    print("Model loaded successfully")

    # Load test dataset
    CVL_ROOT = "../datasets/cvl-database-1-1"
    test_dataset = CVLTestDataset(CVL_ROOT, char2idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Test
    test_model(model, test_loader)
