import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

from htr_model_v2 import (
    HTRModelV2, greedy_decode, beam_decode, compute_cer,
    tensor_to_string, NUM_CLASSES, MAX_DECODE_LEN,
    char2idx, idx2char, EOS_IDX, EMA,
)

# =====================================================
# DEVICE
# =====================================================

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = 64

# =====================================================
# IMAGE LOADING
# =====================================================

def load_image(path):
    img    = Image.open(path).convert("L")
    orig_w, orig_h = img.size
    new_w  = max(32, int(orig_w * IMG_HEIGHT / orig_h))
    img    = img.resize((new_w, IMG_HEIGHT), Image.LANCZOS)
    t      = transforms.ToTensor()(img)
    t      = transforms.Normalize((0.5,), (0.5,))(t)
    return t.unsqueeze(0)

# =====================================================
# MODEL LOADING
# Supports:
#   - raw state dict  (best_htr_model_v2.pth)
#   - full checkpoint (checkpoint_v2.pth)
#   - EMA shadow weights (optional)
# =====================================================

def load_model(checkpoint_path, use_ema=False):
    model = HTRModelV2(NUM_CLASSES).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Full checkpoint dict vs raw state dict
    if "model_state_dict" in state:
        model_state = state["model_state_dict"]
    else:
        model_state = state

    # Handle position_queries size mismatch
    if "decoder.position_queries" in model_state:
        old = model_state["decoder.position_queries"]
        if old.shape[1] != MAX_DECODE_LEN:
            resized = F.interpolate(
                old.permute(0, 2, 1), size=MAX_DECODE_LEN,
                mode='linear', align_corners=False
            )
            model_state["decoder.position_queries"] = resized.permute(0, 2, 1)

    # Handle query_pos_enc.pe size mismatch
    if "decoder.query_pos_enc.pe" in model_state:
        old_pe  = model_state["decoder.query_pos_enc.pe"]
        new_len = model.decoder.query_pos_enc.pe.shape[1]
        if old_pe.shape[1] != new_len:
            resized = F.interpolate(
                old_pe.permute(0, 2, 1), size=new_len,
                mode='linear', align_corners=False
            )
            model_state["decoder.query_pos_enc.pe"] = resized.permute(0, 2, 1)

    model.load_state_dict(model_state, strict=False)

    # Optionally apply EMA shadow weights
    if use_ema:
        ema = EMA(model, decay=0.999)
        ema.apply_shadow()
        print("EMA shadow weights applied.")

    model.eval()
    print(f"Model loaded from '{checkpoint_path}' — {DEVICE}")
    return model

# =====================================================
# CORE METRICS
# =====================================================

def word_error_rate(preds, gts):
    total_edits = 0
    total_words = 0
    for p, g in zip(preds, gts):
        from htr_model_v2 import edit_distance
        pw = p.split()
        gw = g.split()
        total_edits += edit_distance(pw, gw)
        total_words += len(gw)
    return total_edits / max(1, total_words)


def sequence_accuracy(preds, gts):
    correct = sum(1 for p, g in zip(preds, gts) if p.strip() == g.strip())
    return correct / max(1, len(gts))


def normalised_edit_similarity(preds, gts):
    from htr_model_v2 import edit_distance
    scores = []
    for p, g in zip(preds, gts):
        max_len = max(len(p), len(g), 1)
        ned     = edit_distance(p, g) / max_len
        scores.append(1 - ned)
    return np.mean(scores)


def char_precision_recall_f1(preds, gts):
    total_tp = total_fp = total_fn = 0
    for p, g in zip(preds, gts):
        min_len  = min(len(p), len(g))
        tp       = sum(1 for i in range(min_len) if p[i] == g[i])
        total_tp += tp
        total_fp += len(p) - tp
        total_fn += len(g) - tp
    precision = total_tp / max(1, total_tp + total_fp)
    recall    = total_tp / max(1, total_tp + total_fn)
    f1        = (2 * precision * recall) / max(1e-9, precision + recall)
    return precision, recall, f1


def per_character_class_f1(preds, gts, top_n=15):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for p, g in zip(preds, gts):
        min_len = min(len(p), len(g))
        for i in range(min_len):
            if p[i] == g[i]:
                tp[g[i]] += 1
            else:
                fp[p[i]] += 1
                fn[g[i]] += 1
        for i in range(min_len, len(p)):
            fp[p[i]] += 1
        for i in range(min_len, len(g)):
            fn[g[i]] += 1

    all_chars = set(tp) | set(fp) | set(fn)
    per_class = {}
    for c in all_chars:
        p_val = tp[c] / max(1, tp[c] + fp[c])
        r_val = tp[c] / max(1, tp[c] + fn[c])
        f     = (2 * p_val * r_val) / max(1e-9, p_val + r_val)
        per_class[c] = {
            "precision": p_val,
            "recall":    r_val,
            "f1":        f,
            "support":   tp[c] + fn[c],
        }

    macro_f1     = np.mean([v["f1"] for v in per_class.values()])
    sorted_chars = sorted(per_class.items(),
                          key=lambda x: x[1]["support"], reverse=True)
    return macro_f1, sorted_chars[:top_n]


def length_bucket_cer(preds, gts,
                      buckets=((0,10),(10,20),(20,40),(40,70),(70,200))):
    results = {}
    for lo, hi in buckets:
        bp = [p for p, g in zip(preds, gts) if lo <= len(g) < hi]
        bg = [g for g in gts if lo <= len(g) < hi]
        if bg:
            results[f"{lo}-{hi} chars"] = (compute_cer(bp, bg), len(bg))
    return results


def pass_comparison(model, val_loader):
    """Compare Pass 1 vs Pass 2 CER to quantify refinement benefit."""
    p1_preds = []
    p2_preds = []
    gts      = []

    model.eval()
    with torch.no_grad():
        for images, targets, label_lengths in val_loader:
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)
            T       = targets.size(1)

            logits1, logits2 = model(images, num_decode_steps=T)

            p1_preds.extend(greedy_decode(logits1))
            p2_preds.extend(greedy_decode(logits2))
            gts.extend([tensor_to_string(targets[i])
                        for i in range(targets.size(0))])

    cer1 = compute_cer(p1_preds, gts)
    cer2 = compute_cer(p2_preds, gts)
    return cer1, cer2, gts, p1_preds, p2_preds

# =====================================================
# EVALUATE FROM VALIDATION LOADER
# =====================================================

def evaluate_loader(model, val_loader, use_beam=False):
    preds_all = []
    gts_all   = []

    model.eval()
    with torch.no_grad():
        for images, targets, label_lengths in val_loader:
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)
            T       = targets.size(1)

            _, logits2 = model(images, num_decode_steps=T)

            if use_beam:
                preds = beam_decode(logits2, beam_width=5)
            else:
                preds = greedy_decode(logits2)

            gt_strs = [tensor_to_string(targets[i])
                       for i in range(targets.size(0))]

            preds_all.extend(preds)
            gts_all.extend(gt_strs)

    return preds_all, gts_all

# =====================================================
# EVALUATE FROM FOLDER
# labels.txt: one line per image — "filename.png\tground truth"
# =====================================================

def evaluate_folder(model, folder, labels_file, use_beam=False):
    labels = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                parts = line.split("\t", 1)
                labels[parts[0]] = parts[1]

    preds_all = []
    gts_all   = []
    skipped   = 0

    model.eval()
    for fname, gt in labels.items():
        img_path = os.path.join(folder, fname)
        if not os.path.exists(img_path):
            skipped += 1
            continue
        try:
            img_tensor = load_image(img_path).to(DEVICE)
        except Exception:
            skipped += 1
            continue

        with torch.no_grad():
            _, logits2 = model(img_tensor, num_decode_steps=MAX_DECODE_LEN)

        if use_beam:
            pred = beam_decode(logits2, beam_width=5)[0]
        else:
            pred = greedy_decode(logits2)[0]

        preds_all.append(pred)
        gts_all.append(gt)

    return preds_all, gts_all, skipped

# =====================================================
# PRINT REPORT
# =====================================================

def print_report(preds, gts, show_samples=10, top_n_chars=15,
                 pass1_cer=None, pass2_cer=None):
    print("\n" + "="*60)
    print("  HTR EVALUATION REPORT")
    print("="*60)
    print(f"  Total samples : {len(gts)}")

    cer     = compute_cer(preds, gts)
    wer     = word_error_rate(preds, gts)
    seq_acc = sequence_accuracy(preds, gts)
    nes     = normalised_edit_similarity(preds, gts)
    prec, rec, f1 = char_precision_recall_f1(preds, gts)
    macro_f1, per_class = per_character_class_f1(preds, gts, top_n=top_n_chars)

    print(f"\n  ── Sequence-level ──")
    print(f"  Exact Match Accuracy : {seq_acc*100:.2f}%")
    print(f"  CER (↓ better)       : {cer:.4f}  ({cer*100:.2f}%)")
    print(f"  WER (↓ better)       : {wer:.4f}  ({wer*100:.2f}%)")
    print(f"  Norm Edit Similarity : {nes:.4f}  ({nes*100:.2f}%)")

    print(f"\n  ── Character-level ──")
    print(f"  Precision  : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  Recall     : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  F1 Score   : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Macro F1   : {macro_f1:.4f}  ({macro_f1*100:.2f}%)")

    # Pass 1 vs Pass 2 comparison (if available)
    if pass1_cer is not None and pass2_cer is not None:
        improvement = (pass1_cer - pass2_cer) / max(pass1_cer, 1e-6) * 100
        print(f"\n  ── Refinement analysis ──")
        print(f"  Pass 1 CER  : {pass1_cer:.4f}  ({pass1_cer*100:.2f}%)")
        print(f"  Pass 2 CER  : {pass2_cer:.4f}  ({pass2_cer*100:.2f}%)")
        print(f"  Improvement : {improvement:.1f}% relative CER reduction from refinement")

    # CER by length bucket
    print(f"\n  ── CER by sequence length ──")
    buckets = length_bucket_cer(preds, gts)
    for bucket, (b_cer, count) in buckets.items():
        filled = int((1 - min(b_cer, 1.0)) * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"  {bucket:>15}  CER={b_cer:.3f}  n={count:>5}  {bar}")

    # Per-character F1
    print(f"\n  ── Per-character F1 (top {top_n_chars} by frequency) ──")
    print(f"  {'Char':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Support':>8}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")
    for char, stats in per_class:
        display = repr(char) if char in (' ', '\t') else char
        print(f"  {display:>6}  {stats['f1']:>6.3f}  "
              f"{stats['precision']:>6.3f}  {stats['recall']:>6.3f}  "
              f"{stats['support']:>8}")

    # Sample predictions
    if show_samples > 0:
        print(f"\n  ── Sample predictions ──")
        indices = np.linspace(0, len(preds)-1,
                              min(show_samples, len(preds)), dtype=int)
        for i in indices:
            cer_i = compute_cer([preds[i]], [gts[i]])
            gt_display   = gts[i][:80]
            pred_display = preds[i][:80]
            print(f"  GT  : {gt_display}")
            print(f"  Pred: {pred_display}   (CER: {cer_i:.3f})")
            print()

    print("="*60)
    return {
        "exact_match": seq_acc,
        "cer":         cer,
        "wer":         wer,
        "nes":         nes,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "macro_f1":    macro_f1,
        "pass1_cer":   pass1_cer,
        "pass2_cer":   pass2_cer,
    }

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HTR Model v2 — Full Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation split (recommended)
  python evaluate.py --model best_htr_model_v2.pth --use-val-loader

  # Also show Pass 1 vs Pass 2 refinement comparison
  python evaluate.py --model best_htr_model_v2.pth --use-val-loader --pass-compare

  # Evaluate on folder of images with labels file
  python evaluate.py --model best_htr_model_v2.pth --folder images/ --labels labels.txt

  # Use beam search decode (may improve CER slightly)
  python evaluate.py --model best_htr_model_v2.pth --use-val-loader --beam

  # Save metrics to JSON
  python evaluate.py --model best_htr_model_v2.pth --use-val-loader --save metrics.json

  labels.txt format (tab-separated):
    image001.png    Hello World
    image002.png    Tab Pantoprazole 40mg
        """
    )

    parser.add_argument("--model",          required=True)
    parser.add_argument("--folder",         default=None)
    parser.add_argument("--labels",         default=None)
    parser.add_argument("--use-val-loader", action="store_true")
    parser.add_argument("--pass-compare",   action="store_true",
                        help="Show Pass 1 vs Pass 2 CER comparison")
    parser.add_argument("--beam",           action="store_true",
                        help="Use beam search decode (beam width 5)")
    parser.add_argument("--samples",        type=int, default=10)
    parser.add_argument("--top-chars",      type=int, default=15)
    parser.add_argument("--save",           default=None)

    args  = parser.parse_args()
    model = load_model(args.model)

    pass1_cer = pass2_cer = None

    if args.use_val_loader:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_v2 import val_loader
        print(f"Running on validation split — {'beam' if args.beam else 'greedy'} decode...")

        preds, gts = evaluate_loader(model, val_loader, use_beam=args.beam)

        if args.pass_compare:
            print("Running Pass 1 vs Pass 2 comparison...")
            pass1_cer, pass2_cer, _, _, _ = pass_comparison(model, val_loader)

    elif args.folder and args.labels:
        print(f"Running on folder: {args.folder}")
        preds, gts, skipped = evaluate_folder(
            model, args.folder, args.labels, use_beam=args.beam
        )
        if skipped:
            print(f"  Skipped {skipped} images")

    else:
        parser.error("Provide --use-val-loader OR --folder + --labels")

    metrics = print_report(
        preds, gts,
        show_samples=args.samples,
        top_n_chars=args.top_chars,
        pass1_cer=pass1_cer,
        pass2_cer=pass2_cer,
    )

    if args.save:
        import json
        with open(args.save, "w") as f:
            json.dump({k: float(v) if v is not None else None
                       for k, v in metrics.items()}, f, indent=2)
        print(f"\nMetrics saved to '{args.save}'")