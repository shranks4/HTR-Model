import math
import torch
import string
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# CHARACTER SET
# =====================================================

characters = string.ascii_letters + string.digits + " .,-()/:"

char2idx = {char: idx + 1 for idx, char in enumerate(characters)}
char2idx["<blank>"] = 0
char2idx["<eos>"]   = len(char2idx)
char2idx["<sos>"]   = len(char2idx)

idx2char = {idx: char for char, idx in char2idx.items()}

NUM_CLASSES    = len(char2idx)
SOS_IDX        = char2idx["<sos>"]
EOS_IDX        = char2idx["<eos>"]
MAX_DECODE_LEN = 96


# =====================================================
# POSITIONAL ENCODING
# =====================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# =====================================================
# NOVELTY #1: MULTI-SCALE CNN BACKBONE
# =====================================================

class MultiScaleCNNBackbone(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        self.stage1 = nn.Sequential(
            self._conv_block(1, 64),
            nn.MaxPool2d(2, 2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2, 2),
        )
        self.stage2 = nn.Sequential(
            self._conv_block(128, 256),
        )
        self.stage3 = nn.Sequential(
            self._conv_block(256, 512),
            nn.MaxPool2d((2, 1)),
        )

        self.proj1 = nn.Conv2d(128, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(512, embed_dim, kernel_size=1)

        self.scale_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        p1 = self.proj1(s1)
        p2 = self.proj2(s2)
        p3 = self.proj3(s3)

        target_h, target_w = p3.shape[2], p3.shape[3]
        p1 = F.adaptive_avg_pool2d(p1, (target_h, target_w))
        p2 = F.adaptive_avg_pool2d(p2, (target_h, target_w))

        w     = F.softmax(self.scale_weights, dim=0)
        fused = w[0] * p1 + w[1] * p2 + w[2] * p3
        fused = F.adaptive_avg_pool2d(fused, (1, target_w))
        fused = fused.squeeze(2).permute(0, 2, 1)

        return fused


# =====================================================
# NOVELTY #2: GLYPH-AWARE STROKE EMBEDDINGS
# =====================================================

class GlyphAwareStrokeEmbedding(nn.Module):
    def __init__(self, embed_dim=512, num_angle_bins=8):
        super().__init__()
        self.num_angle_bins = num_angle_bins
        self.embed_dim      = embed_dim

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        self.stroke_proj = nn.Sequential(
            nn.Linear(num_angle_bins + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.stroke_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _soft_angle_histogram(self, angle, magnitude, width):
        B = angle.shape[0]
        H = angle.shape[2]

        angle_norm = (angle + math.pi) / (2 * math.pi)
        bins       = (angle_norm * self.num_angle_bins).long().clamp(0, self.num_angle_bins - 1)
        bins       = F.interpolate(bins.float(), size=(H, width), mode="nearest").long()
        magnitude  = F.interpolate(magnitude, size=(H, width), mode="bilinear", align_corners=False)

        hist = torch.zeros(B, width, self.num_angle_bins, device=angle.device, dtype=torch.float32)
        for b_idx in range(B):
            for w_idx in range(width):
                col_bins = bins[b_idx, 0, :, w_idx]
                col_mag  = magnitude[b_idx, 0, :, w_idx]
                hist[b_idx, w_idx].scatter_add_(0, col_bins, col_mag)

        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)
        return hist

    def forward(self, x_img, seq_len):
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(1)

        gx        = F.conv2d(x_img, self.sobel_x, padding=1)
        gy        = F.conv2d(x_img, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        angle     = torch.atan2(gy, gx)

        hist        = self._soft_angle_histogram(angle, magnitude, seq_len)
        mag_resized = F.adaptive_avg_pool2d(magnitude, (1, seq_len))
        mag_col     = mag_resized.squeeze(1).squeeze(1).unsqueeze(-1)

        stroke_feat = torch.cat([hist, mag_col], dim=-1)
        return self.stroke_proj(stroke_feat)


# =====================================================
# NOVELTY #3: QUERY-BASED DECODER WITH GATED REFINEMENT
#
# FIX: Causal mask added to _decode().
# Without it, position 5 attends to the same visual
# memory as position 25 with no sequential constraint,
# causing attention entropy collapse at long positions
# and the consistent failure after character 15-20.
# =====================================================

class QueryBasedDecoder(nn.Module):
    def __init__(self, num_classes, embed_dim=512, max_len=MAX_DECODE_LEN,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.max_len   = max_len
        self.embed_dim = embed_dim

        self.position_queries = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.01
        )
        self.query_pos_enc = PositionalEncoding(embed_dim)

        self.char_embed = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.char_embed.weight, mean=0.0, std=0.02)

        self.refinement_temp = nn.Parameter(torch.tensor(1.0))

        self.refinement_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.decoder    = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout    = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

        for m in self.refinement_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _decode(self, queries, memory):
        T = queries.size(1)

        # Causal mask: position i cannot attend to position j > i.
        # This enforces sequential coherence — each output position
        # builds on previous positions rather than attending to the
        # full sequence simultaneously, which caused the collapse
        # after character 15-20 in earlier versions.
        causal_mask = torch.triu(
            torch.ones(T, T, device=queries.device), diagonal=1
        ).bool()

        out    = self.decoder(queries, memory, tgt_mask=causal_mask)
        out    = self.dropout(out)
        logits = self.classifier(out)
        return logits

    def forward(self, memory, num_steps=None):
        B = memory.size(0)
        T = min(num_steps if num_steps else self.max_len, self.max_len)

        # Pass 1
        queries = self.position_queries[:, :T, :].expand(B, -1, -1)
        queries = self.query_pos_enc(queries)
        logits1 = self._decode(queries, memory)

        # Pass 2: gated refinement
        # .detach() prevents gradients flowing back through Pass 1
        # which would freeze the refinement gap
        temp       = torch.abs(self.refinement_temp) + 0.5
        probs      = F.softmax(logits1.detach() / temp, dim=-1)
        soft_embed = probs @ self.char_embed.weight

        gate      = self.refinement_gate(
            torch.cat([queries, soft_embed], dim=-1)
        )
        refined_q = queries + gate * soft_embed
        logits2   = self._decode(refined_q, memory)

        return logits1, logits2


# =====================================================
# LOSS
# =====================================================

class HTRLoss(nn.Module):
    def __init__(self, num_classes, aux_weight=0.3, consistency_weight=0.05):
        super().__init__()
        self.aux_weight         = aux_weight
        self.consistency_weight = consistency_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    def forward(self, logits1, logits2, targets, scale_weights=None):
        B, T, C = logits2.shape

        primary_loss = self.ce(logits2.reshape(B * T, C), targets.reshape(B * T))
        aux_loss     = self.ce(logits1.reshape(B * T, C), targets.reshape(B * T))

        p1               = F.log_softmax(logits1, dim=-1)
        p2               = F.softmax(logits2, dim=-1)
        reverse_kl       = F.kl_div(p1, p2, reduction="batchmean", log_target=False)
        forward_kl       = F.kl_div(p2.log(), p1.exp(), reduction="batchmean")
        consistency_loss = 0.3 * forward_kl + 0.7 * reverse_kl

        total = (
            primary_loss
            + self.aux_weight * aux_loss
            + self.consistency_weight * consistency_loss
        )

        scale_reg = torch.tensor(0.0, device=logits2.device)
        if scale_weights is not None:
            sw        = F.softmax(scale_weights, dim=0)
            entropy   = -(sw * torch.log(sw + 1e-6)).sum()
            scale_reg = 0.05 * (math.log(3) - entropy)
            total     = total + scale_reg

        return total, {
            "primary":     primary_loss.item(),
            "auxiliary":   aux_loss.item(),
            "consistency": consistency_loss.item(),
            "scale_reg":   scale_reg.item(),
        }


# =====================================================
# FULL HTR MODEL v2
# =====================================================

class HTRModelV2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        self.cnn             = MultiScaleCNNBackbone(embed_dim)
        self.stroke_embed    = GlyphAwareStrokeEmbedding(embed_dim)
        self.pos_encoder     = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = QueryBasedDecoder(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, x, num_decode_steps=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        vis_feat    = self.cnn(x)
        S           = vis_feat.size(1)
        stroke_feat = self.stroke_embed(x, S)

        features = vis_feat + stroke_feat
        features = self.pos_encoder(features)
        memory   = self.transformer_encoder(features)

        logits1, logits2 = self.decoder(memory, num_decode_steps)
        return logits1, logits2


# =====================================================
# EMA
# =====================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg           = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data        = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def tensor_to_string(label_tensor):
    chars = []
    for idx in label_tensor.tolist():
        if idx in (SOS_IDX, EOS_IDX, char2idx["<blank>"], -1, 0):
            continue
        if idx in idx2char:
            chars.append(idx2char[idx])
    return "".join(chars)


def greedy_decode(logits2):
    preds   = logits2.argmax(dim=-1)
    results = []
    for seq in preds:
        chars = []
        for idx in seq.tolist():
            if idx == EOS_IDX:
                break
            if idx in idx2char and idx not in (SOS_IDX, char2idx["<blank>"]):
                chars.append(idx2char[idx])
        results.append("".join(chars))
    return results


def beam_decode(logits2, beam_width=5):
    B, T, C   = logits2.shape
    log_probs = F.log_softmax(logits2, dim=-1)
    results   = []

    for b in range(B):
        beams = [(0.0, [])]
        for t in range(T):
            new_beams = []
            for score, tokens in beams:
                top_probs, top_ids = log_probs[b, t].topk(beam_width)
                for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
                    new_beams.append((score + prob, tokens + [idx]))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        best_tokens = beams[0][1]
        chars = []
        for idx in best_tokens:
            if idx == EOS_IDX:
                break
            if idx not in (SOS_IDX, char2idx["<blank>"], -1, 0):
                chars.append(idx2char.get(idx, ""))
        results.append("".join(chars))

    return results


def edit_distance(s1, s2):
    dp = np.zeros((len(s1) + 1, len(s2) + 1))
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost     = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(s1)][len(s2)]


def compute_cer(preds, gts):
    total_edits = 0
    total_chars = 0
    for p, g in zip(preds, gts):
        total_edits += edit_distance(p, g)
        total_chars += len(g)
    return total_edits / total_chars if total_chars > 0 else 0.0


# =====================================================
# READY
# =====================================================

if __name__ == "__main__":
    model   = HTRModelV2(NUM_CLASSES).to(DEVICE)
    loss_fn = HTRLoss(NUM_CLASSES).to(DEVICE)

    B, H, W  = 2, 64, 256
    x        = torch.randn(B, 1, H, W).to(DEVICE)
    targets  = torch.randint(1, NUM_CLASSES - 2, (B, MAX_DECODE_LEN)).to(DEVICE)

    logits1, logits2 = model(x)
    loss, breakdown  = loss_fn(logits1, logits2, targets, model.cnn.scale_weights)

    print("HTR model v2 ready.")
    print("Output shape :", logits2.shape)
    print("Loss         :", round(loss.item(), 4))
    print("Breakdown    :", breakdown)
    print("Scale weights:", F.softmax(model.cnn.scale_weights, dim=0).detach())