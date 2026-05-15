import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

from htr_model_v2 import (
    HTRModelV2, beam_decode, greedy_decode,
    NUM_CLASSES, MAX_DECODE_LEN,
    char2idx, idx2char, EOS_IDX
)

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# IMAGE PREPROCESSING
# Must match what your dataset_loader does during training.
# Adjust IMG_HEIGHT if your training used a different height.
# =====================================================

IMG_HEIGHT = 64

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_HEIGHT * 8)),   # H fixed, W stretched
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def load_image(path):
    """Load and preprocess a single image. Returns (1, 1, H, W) tensor."""
    img = Image.open(path).convert("L")

    # Preserve aspect ratio — resize height to IMG_HEIGHT, scale width accordingly
    orig_w, orig_h = img.size
    new_w = max(32, int(orig_w * IMG_HEIGHT / orig_h))
    img = img.resize((new_w, IMG_HEIGHT), Image.LANCZOS)

    # Convert to tensor and normalise
    t = transforms.ToTensor()(img)                    # (1, H, W)
    t = transforms.Normalize((0.5,), (0.5,))(t)

    return t.unsqueeze(0)                             # (1, 1, H, W)


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(checkpoint_path):
    model = HTRModelV2(NUM_CLASSES).to(DEVICE)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at '{checkpoint_path}'")
        sys.exit(1)

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Support both raw state_dict and full checkpoint dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    # Handle position_queries size mismatch
    if "decoder.position_queries" in state:
        old = state["decoder.position_queries"]
        new_shape = (1, MAX_DECODE_LEN, old.shape[2])
        if old.shape != torch.Size(new_shape):
            resized = F.interpolate(
                old.permute(0, 2, 1),
                size=MAX_DECODE_LEN,
                mode='linear',
                align_corners=False
            )
            state["decoder.position_queries"] = resized.permute(0, 2, 1)

    # Handle query_pos_enc.pe size mismatch
    if "decoder.query_pos_enc.pe" in state:
        old_pe  = state["decoder.query_pos_enc.pe"]
        new_len = model.decoder.query_pos_enc.pe.shape[1]
        if old_pe.shape[1] != new_len:
            resized = F.interpolate(
                old_pe.permute(0, 2, 1),
                size=new_len,
                mode='linear',
                align_corners=False
            )
            state["decoder.query_pos_enc.pe"] = resized.permute(0, 2, 1)

    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded from '{checkpoint_path}' — running on {DEVICE}")
    return model


# =====================================================
# PREDICT
# =====================================================

def predict(model, image_tensor, beam_width=5):
    """
    image_tensor : (1, 1, H, W)
    Returns: (text, confidence, per_char_probs)
    """
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        _, logits2 = model(image_tensor, num_decode_steps=MAX_DECODE_LEN)

    # Beam decode for best text
    texts = beam_decode(logits2, beam_width=beam_width)
    text  = texts[0]

    # Per-character confidence from greedy (cleaner than beam for confidence)
    probs     = F.softmax(logits2[0], dim=-1)         # (T, C)
    top_probs = probs.max(dim=-1).values              # (T,)
    top_ids   = probs.argmax(dim=-1)                  # (T,)

    per_char = []
    for t, (idx, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist())):
        if idx == EOS_IDX:
            break
        if idx in idx2char and idx not in (char2idx.get("<blank>", -1),
                                            char2idx.get("<sos>", -1)):
            per_char.append((idx2char[idx], prob))

    # Overall confidence = geometric mean of per-char probs
    if per_char:
        import math
        log_conf = sum(math.log(p + 1e-9) for _, p in per_char) / len(per_char)
        confidence = math.exp(log_conf)
    else:
        confidence = 0.0

    return text, confidence, per_char


# =====================================================
# DISPLAY HELPERS
# =====================================================

def print_result(path, text, confidence, per_char, verbose=False):
    print(f"\n{'─'*60}")
    print(f"  File      : {os.path.basename(path)}")
    print(f"  Prediction: {text}")
    print(f"  Confidence: {confidence*100:.1f}%")

    if verbose and per_char:
        print(f"  Per-char  :", end=" ")
        for char, prob in per_char:
            bar = "█" * int(prob * 10)
            print(f"'{char}'({prob:.2f})", end=" ")
        print()


# =====================================================
# SUPPORTED IMAGE EXTENSIONS
# =====================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(
        description="HTR Model v2 — Inference",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Single image
  python infer.py --model best_htr_model_v2.pth --image path/to/image.png

  # Folder of images
  python infer.py --model best_htr_model_v2.pth --folder path/to/images/

  # Verbose (show per-character confidence)
  python infer.py --model best_htr_model_v2.pth --image img.png --verbose

  # Save results to file
  python infer.py --model best_htr_model_v2.pth --folder images/ --output results.txt

  # Use greedy decode instead of beam search (faster)
  python infer.py --model best_htr_model_v2.pth --image img.png --greedy

  # Adjust beam width (default 5)
  python infer.py --model best_htr_model_v2.pth --image img.png --beam 10
        """
    )

    parser.add_argument("--model",   required=True,
                        help="Path to best_htr_model_v2.pth or checkpoint_v2.pth")
    parser.add_argument("--image",   default=None,
                        help="Path to a single image file")
    parser.add_argument("--folder",  default=None,
                        help="Path to a folder of images")
    parser.add_argument("--output",  default=None,
                        help="Save predictions to this text file")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-character confidence scores")
    parser.add_argument("--greedy",  action="store_true",
                        help="Use greedy decode instead of beam search")
    parser.add_argument("--beam",    type=int, default=5,
                        help="Beam width for beam search (default: 5)")

    args = parser.parse_args()

    if args.image is None and args.folder is None:
        parser.error("Provide --image or --folder")

    # Load model
    model = load_model(args.model)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.folder:
        for fname in sorted(os.listdir(args.folder)):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                image_paths.append(os.path.join(args.folder, fname))

    if not image_paths:
        print("No images found.")
        sys.exit(0)

    print(f"\nRunning inference on {len(image_paths)} image(s)...")

    results = []

    for path in image_paths:
        try:
            img_tensor = load_image(path)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue

        if args.greedy:
            with torch.no_grad():
                _, logits2 = model(img_tensor.to(DEVICE),
                                   num_decode_steps=MAX_DECODE_LEN)
            texts      = greedy_decode(logits2)
            text       = texts[0]
            confidence = 0.0
            per_char   = []
        else:
            text, confidence, per_char = predict(model, img_tensor,
                                                  beam_width=args.beam)

        print_result(path, text, confidence, per_char, verbose=args.verbose)
        results.append((path, text, confidence))

    # Summary
    print(f"\n{'─'*60}")
    print(f"  Done. {len(results)}/{len(image_paths)} images processed.")

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for path, text, conf in results:
                f.write(f"{os.path.basename(path)}\t{text}\t{conf*100:.1f}%\n")
        print(f"  Results saved to '{args.output}'")


if __name__ == "__main__":
    main()