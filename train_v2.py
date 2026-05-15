import torch
import os
import math
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, random_split
import torch.nn.functional as F

from dataset_loader import (
    IAMDataset,
    CVLWordsDataset,
    RIMESDataset,
    MedicalPrescriptionDataset,
    BDDataset,
)

from htr_model_v2 import (
    HTRModelV2, HTRLoss, greedy_decode, compute_cer,
    tensor_to_string, NUM_CLASSES, MAX_DECODE_LEN,
    char2idx, idx2char, SOS_IDX, EOS_IDX, EMA,
)

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =====================================================
# DATASET PATHS
# =====================================================

IAM_ROOT   = "../datasets/iam"
CVL_ROOT   = "../datasets/cvl-database-1-1/cvl-database-1-1"
RIMES_ROOT = "../datasets/Handwritten2Text Training Dataset"
BD_ROOT    = "../datasets/bd-dataset"

# =====================================================
# LOAD DATASETS
# =====================================================

print("\nLoading CVL dataset...")
cvl_dataset = CVLWordsDataset(CVL_ROOT, char2idx)

print("\nLoading RIMES dataset...")
rimes_dataset = RIMESDataset(RIMES_ROOT, char2idx)

print("\nLoading IAM dataset...")
iam_dataset = IAMDataset(IAM_ROOT, char2idx)


print("\nLoading BD dataset...")
bd_dataset = BDDataset(BD_ROOT, char2idx, splits=("train", "val"))

# =====================================================
# VALIDATION SPLIT
# =====================================================

combined_dataset = ConcatDataset([
    cvl_dataset, rimes_dataset, iam_dataset, bd_dataset
])

total_samples = len(combined_dataset)
val_size      = int(0.1 * total_samples)
train_size    = total_samples - val_size

train_dataset, val_dataset_split = random_split(
    combined_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# =====================================================
# LENGTH-STRATIFIED SAMPLING
#
# FIX: Upweight long-sequence datasets (RIMES, IAM)
# and downweight short-sequence datasets (BD, CVL).
# Root cause of the 15-char collapse: BD+CVL were
# ~40% of data, all under 12 chars. Position queries
# at index 15+ had almost no supervision signal.
#
# Source weights:
#   CVL   — word-level (3-8 chars)   → 1.0x
#   RIMES — line-level (30-60 chars) → 3.0x  (upweight)
#   IAM   — line-level (20-50 chars) → 2.5x  (upweight)
#   MED   — line-level (10-40 chars) → 2.0x
#   BD    — word-level (5-12 chars)  → 0.5x  (downweight)
# =====================================================

dataset_sizes   = [
    len(cvl_dataset),
    len(rimes_dataset),
    len(iam_dataset),
    len(bd_dataset),
]
source_weights  = [1.0, 3.0, 2.5, 2.0, 0.5]

all_weights = []
for size, w in zip(dataset_sizes, source_weights):
    all_weights.extend([w] * size)

# Map back to train_dataset indices (random_split reorders)
train_weights = [all_weights[train_dataset.indices[i]]
                 for i in range(len(train_dataset))]

sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

print("\nSource weights applied:")
for name, size, w in zip(
    ["CVL", "RIMES", "IAM", "MED", "BD"],
    dataset_sizes, source_weights
):
    print(f"  {name:6s}: {size:6d} samples × {w:.1f}x")

# =====================================================
# COLLATE FUNCTION
# Dynamic batch length — targets trimmed to longest
# GT in the batch, not global MAX_DECODE_LEN.
# =====================================================

def collate_fn_v2(batch):
    images        = []
    label_list    = []
    label_lengths = []

    for img, label, width in batch:
        images.append(img)
        label_ints = label.tolist() if isinstance(label, torch.Tensor) else list(label)
        label_ints = label_ints + [EOS_IDX]
        label_list.append(label_ints)
        label_lengths.append(len(label_ints))

    images        = torch.stack(images)
    B             = len(batch)
    batch_max_len = min(max(label_lengths), MAX_DECODE_LEN)

    targets = torch.full((B, batch_max_len), -1, dtype=torch.long)
    for i, label_ints in enumerate(label_list):
        length = min(len(label_ints), batch_max_len)
        targets[i, :length] = torch.tensor(label_ints[:length], dtype=torch.long)

    return images, targets, torch.tensor(label_lengths, dtype=torch.long)

# =====================================================
# DATALOADER
# =====================================================

BATCH_SIZE  = 4
ACCUM_STEPS = 4   # effective batch = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_fn_v2,
)

val_loader = DataLoader(
    val_dataset_split,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn_v2,
)

# =====================================================
# MODEL & LOSS
# =====================================================

model   = HTRModelV2(NUM_CLASSES).to(DEVICE)
loss_fn = HTRLoss(NUM_CLASSES, aux_weight=0.3, consistency_weight=0.01).to(DEVICE)

# =====================================================
# OPTIMIZER
# Decoder LR 2x base — position queries need more
# gradient signal as the least-trained component.
# Gate has higher weight decay to prevent saturation.
# =====================================================

gate_params   = list(model.decoder.refinement_gate.parameters())
gate_ids      = set(id(p) for p in gate_params)
decoder_params = [p for p in model.decoder.parameters() if id(p) not in gate_ids]
decoder_ids   = set(id(p) for p in model.decoder.parameters())
base_params   = [p for p in model.parameters() if id(p) not in decoder_ids]

optimizer = torch.optim.AdamW(
    [
        {"params": base_params,    "lr": 1e-4,  "weight_decay": 1e-4},
        {"params": decoder_params, "lr": 2e-4,  "weight_decay": 1e-4},
        {"params": gate_params,    "lr": 2e-4,  "weight_decay": 2e-3},
    ],
    weight_decay=1e-4,
)

# =====================================================
# SCHEDULER: Linear warmup → Cosine decay
# =====================================================

EPOCHS       = 25
WARMUP_STEPS = 5000
TOTAL_STEPS  = (len(train_loader) // ACCUM_STEPS) * EPOCHS

def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return current_step / max(1, WARMUP_STEPS)
    progress = (current_step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return max(0.05, 0.5 * (1.0 + math.cos(progress * math.pi)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =====================================================
# EMA
# =====================================================

ema = EMA(model, decay=0.999)

# =====================================================
# CHECKPOINTING
# =====================================================

CHECKPOINT_PATH = "checkpoint_v2.pth"
start_epoch     = 0
best_val_cer    = float("inf")
global_step     = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    model_state = checkpoint["model_state_dict"]

    if "decoder.position_queries" in model_state:
        old_queries = model_state["decoder.position_queries"]
        new_shape   = (1, MAX_DECODE_LEN, old_queries.shape[2])
        if old_queries.shape != torch.Size(new_shape):
            print(f"Resizing position_queries: {old_queries.shape} → {new_shape}")
            resized = F.interpolate(
                old_queries.permute(0, 2, 1),
                size=MAX_DECODE_LEN,
                mode='linear',
                align_corners=False
            )
            model_state["decoder.position_queries"] = resized.permute(0, 2, 1)

    if "decoder.query_pos_enc.pe" in model_state:
        old_pe  = model_state["decoder.query_pos_enc.pe"]
        new_len = model.decoder.query_pos_enc.pe.shape[1]
        if old_pe.shape[1] != new_len:
            print(f"Resizing query_pos_enc.pe: {old_pe.shape} → (1, {new_len}, 512)")
            resized = F.interpolate(
                old_pe.permute(0, 2, 1),
                size=new_len,
                mode='linear',
                align_corners=False
            )
            model_state["decoder.query_pos_enc.pe"] = resized.permute(0, 2, 1)

    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        print("Warning: partial load (architecture mismatch)")
        result = model.load_state_dict(model_state, strict=False)
        if result.missing_keys:
            print(f"  Missing: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  Unexpected: {result.unexpected_keys}")

    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except Exception as e:
        print(f"Warning: optimizer state not loaded. Fresh start.\n  {e}")
        optimizer.state.clear()

    try:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except Exception as e:
        print(f"Warning: scheduler state not loaded. Fresh start.\n  {e}")

    start_epoch  = checkpoint.get("epoch", 0)
    best_val_cer = checkpoint.get("best_val_cer", float("inf"))
    global_step  = checkpoint.get("global_step", 0)
    print(f"Resumed from epoch {start_epoch}, step {global_step}, best CER {best_val_cer:.4f}")

    ema.register()

    # Restore LR to correct position based on global_step.
    # Needed when optimizer state cannot be loaded (e.g. param group
    # count changed). Without this LR stays at 0 for the whole epoch.
    if global_step > 0:
        if global_step < WARMUP_STEPS:
            restore_factor = global_step / max(1, WARMUP_STEPS)
        else:
            progress       = (global_step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
            restore_factor = max(0.05, 0.5 * (1.0 + math.cos(progress * math.pi)))
        base_lr_values = [1e-4, 2e-4, 2e-4]  # base, decoder, gate
        for pg, base_lr in zip(optimizer.param_groups, base_lr_values):
            pg["lr"] = base_lr * restore_factor
        print(f"LR restored: factor={restore_factor:.4f} -> base={optimizer.param_groups[0]['lr']:.2e} decoder={optimizer.param_groups[1]['lr']:.2e}")

# =====================================================
# LABEL LENGTH ANALYSIS
# =====================================================

def analyze_label_lengths(loader, n_samples=5000):
    print("\n" + "="*40)
    print("Analyzing label length distribution...")
    print("="*40)
    lengths = []
    for _, targets, label_lengths in loader:
        lengths.extend(label_lengths.tolist())
        if len(lengths) >= n_samples:
            break
    lengths = np.array(lengths)
    short   = (lengths < 15).sum()
    print(f"Mean length : {lengths.mean():.1f}")
    print(f"Median      : {np.median(lengths):.1f}")
    print(f"90th pct    : {np.percentile(lengths, 90):.1f}")
    print(f"Max         : {lengths.max():.1f}")
    print(f"Under 15 chars: {short}/{len(lengths)} ({100*short/len(lengths):.1f}%)")
    print("="*40 + "\n")

# =====================================================
# MIXUP AUGMENTATION (image space only)
# =====================================================

def mixup_data(x, alpha=0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, index, lam

# =====================================================
# VALIDATION
# =====================================================

def validate_model(model, val_loader, loss_fn, show_samples=3):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_gts    = []

    with torch.no_grad():
        for images, targets, label_lengths in val_loader:
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)

            T = targets.size(1)
            logits1, logits2 = model(images, num_decode_steps=T)
            loss, _ = loss_fn(logits1, logits2, targets, model.cnn.scale_weights)

            preds   = greedy_decode(logits2)
            gt_strs = [tensor_to_string(targets[i]) for i in range(targets.size(0))]

            all_preds.extend(preds)
            all_gts.extend(gt_strs)
            total_loss += loss.item()

    val_cer = compute_cer(all_preds, all_gts)

    print()
    for p, g in zip(all_preds[:show_samples], all_gts[:show_samples]):
        sample_cer = compute_cer([p], [g])
        print(f"    GT  : '{g}'")
        print(f"    Pred: '{p}'  (CER: {sample_cer:.3f})")
        print()

    w = F.softmax(model.cnn.scale_weights, dim=0)
    print(f"  Scale weights: coarse={w[0]:.3f} mid={w[1]:.3f} fine={w[2]:.3f}")
    print(f"  Refinement temp: {model.decoder.refinement_temp.item():.3f}")
    gate_w = model.decoder.refinement_gate[0].weight
    print(f"  Gate weight norm: {gate_w.norm().item():.4f}")

    return total_loss / max(1, len(val_loader)), val_cer


def validate_with_ema(model, val_loader, loss_fn, ema):
    ema.apply_shadow()
    val_loss, val_cer = validate_model(model, val_loader, loss_fn)
    ema.restore()
    return val_loss, val_cer

# =====================================================
# TRAINING LOOP
# =====================================================

if __name__ == "__main__":

    analyze_label_lengths(train_loader)

    nan_total  = 0
    MIXUP_PROB = 0.3

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        total_loss        = 0
        total_primary     = 0
        total_aux         = 0
        total_consistency = 0
        total_scale_reg   = 0
        nan_batches       = 0
        logged_batches    = 0

        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{EPOCHS}  |  LR(base)={optimizer.param_groups[0]['lr']:.2e}  |  LR(decoder)={optimizer.param_groups[1]['lr']:.2e}")
        print(f"{'='*40}")

        optimizer.zero_grad()

        for batch_idx, (images, targets, label_lengths) in enumerate(train_loader):
            images  = images.to(DEVICE)
            targets = targets.to(DEVICE)
            T       = targets.size(1)

            use_mixup = (np.random.random() < MIXUP_PROB) and (images.size(0) > 1)

            if use_mixup:
                images_mixed, mix_index, lam = mixup_data(images, alpha=0.2)
                logits1, logits2 = model(images_mixed, num_decode_steps=T)
                loss_a, bd_a = loss_fn(logits1, logits2, targets,            model.cnn.scale_weights)
                loss_b, bd_b = loss_fn(logits1, logits2, targets[mix_index], model.cnn.scale_weights)
                loss      = lam * loss_a + (1 - lam) * loss_b
                breakdown = {k: lam * bd_a[k] + (1 - lam) * bd_b[k] for k in bd_a}
            else:
                logits1, logits2 = model(images, num_decode_steps=T)
                loss, breakdown  = loss_fn(logits1, logits2, targets, model.cnn.scale_weights)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
                nan_batches += 1
                optimizer.zero_grad()
                continue

            scaled_loss = loss / ACCUM_STEPS
            scaled_loss.backward()

            total_loss        += loss.item()
            total_primary     += breakdown["primary"]
            total_aux         += breakdown["auxiliary"]
            total_consistency += breakdown["consistency"]
            total_scale_reg   += breakdown["scale_reg"]
            logged_batches    += 1

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                ema.update()
                optimizer.zero_grad()
                global_step += 1

            if batch_idx % 200 == 0:
                print(
                    f"  Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f} "
                    f"(Primary: {breakdown['primary']:.4f}, "
                    f"Aux: {breakdown['auxiliary']:.4f}, "
                    f"Cons: {breakdown['consistency']:.4f}) "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        denom     = max(1, logged_batches)
        avg_loss  = total_loss / denom
        avg_p     = total_primary / denom
        avg_a     = total_aux / denom
        avg_c     = total_consistency / denom
        avg_sr    = total_scale_reg / denom
        nan_total += nan_batches

        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Avg Loss   : {avg_loss:.4f}")
        print(f"  Primary    : {avg_p:.4f}  Aux: {avg_a:.4f}  Cons: {avg_c:.4f}  ScaleReg: {avg_sr:.4f}")
        print(f"  NaN batches: {nan_batches} (total so far: {nan_total})")
        print(f"  Refinement gap (Aux - Primary): {avg_a - avg_p:+.4f}")

        print("  Running validation (EMA weights)...")
        val_loss, val_cer = validate_with_ema(model, val_loader, loss_fn, ema)
        print(f"  Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), "best_htr_model_v2.pth")
            print(f"  ✓ New best model saved (Val CER: {best_val_cer:.4f})")

        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_cer":         best_val_cer,
            "global_step":          global_step,
        }, CHECKPOINT_PATH)

        print("  Checkpoint saved.")

    torch.save(model.state_dict(), "final_htr_model_v2.pth")
    print(f"\nTraining complete. Best Val CER: {best_val_cer:.4f}")