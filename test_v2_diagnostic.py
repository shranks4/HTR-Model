import torch
from htr_model_v2 import HTRModelV2, HTRLoss, NUM_CLASSES, MAX_DECODE_LEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Test model initialization
print("\n" + "="*60)
print("Testing HTRModelV2 initialization...")
print("="*60)

model = HTRModelV2(NUM_CLASSES).to(DEVICE)
loss_fn = HTRLoss(NUM_CLASSES).to(DEVICE)

print("✓ Model and loss function initialized")

# Test forward pass with dummy data
print("\n" + "="*60)
print("Testing forward pass with dummy data...")
print("="*60)

B, H, W = 2, 64, 256
dummy_images = torch.randn(B, 1, H, W).to(DEVICE)
dummy_targets = torch.randint(1, NUM_CLASSES - 2, (B, MAX_DECODE_LEN)).to(DEVICE)

print(f"Input shape: {dummy_images.shape}")
print(f"Targets shape: {dummy_targets.shape}")

with torch.no_grad():
    logits1, logits2 = model(dummy_images, num_decode_steps=MAX_DECODE_LEN)

print(f"Logits1 shape: {logits1.shape} | Contains NaN: {torch.isnan(logits1).any()}")
print(f"Logits2 shape: {logits2.shape} | Contains NaN: {torch.isnan(logits2).any()}")

if not torch.isnan(logits1).any() and not torch.isnan(logits2).any():
    print("✓ Forward pass successful")
else:
    print("✗ NaN detected in forward pass!")
    exit(1)

# Test loss computation
print("\n" + "="*60)
print("Testing loss computation...")
print("="*60)

loss, breakdown = loss_fn(logits1, logits2, dummy_targets)

print(f"Primary loss: {breakdown['primary']:.4f} | NaN: {torch.isnan(torch.tensor(breakdown['primary']))}")
print(f"Auxiliary loss: {breakdown['auxiliary']:.4f} | NaN: {torch.isnan(torch.tensor(breakdown['auxiliary']))}")
print(f"Consistency loss: {breakdown['consistency']:.4f} | NaN: {torch.isnan(torch.tensor(breakdown['consistency']))}")
print(f"Total loss: {loss.item():.4f} | NaN: {torch.isnan(loss)}")

if not torch.isnan(loss):
    print("✓ Loss computation successful")
else:
    print("✗ NaN detected in loss!")
    exit(1)

# Test backward pass
print("\n" + "="*60)
print("Testing backward pass...")
print("="*60)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer.zero_grad()
loss.backward()

# Check for NaN gradients
nan_grads = False
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"✗ NaN gradient in {name}")
        nan_grads = True

if not nan_grads:
    print("✓ Backward pass successful (no NaN gradients)")
else:
    exit(1)

# Optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
optimizer.step()

print("✓ Optimizer step successful")

print("\n" + "="*60)
print("ALL DIAGNOSTICS PASSED ✓")
print("="*60)
