# 4-Direction Execution Plan: Symbolic Alignment Pivots

## **Ranking Overview**

| Direction | Feasibility | Impact | Time | Venue Potential | Recommended Order |
|-----------|------------|--------|------|-----------------|-------------------|
| **Direction 4: Multi-Modal** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4 weeks | CVPR/ICCV | **1st (DO THIS)** |
| **Direction 3: Stratified** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 2 weeks | MICCAI | **2nd (fallback)** |
| **Direction 2: Equivariant** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 6 weeks | CVPR/ICCV | 3rd (if time) |
| **Direction 1: Topological** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 8 weeks | NeurIPS | 4th (ambitious) |

---

## **DIRECTION 4: Multi-Modal Symbolic Alignment** ⭐ PRIMARY RECOMMENDATION

### **Core Innovation**
Learn descriptors that fuse **mask structure** (geometric) with **image appearance** (texture/color) for robust symbolic alignment under appearance shifts.

### **Why This Fixes Your Current Failure**

**Your current problem:**
```python
# Mask-only encoder on corrupted images
mask_corrupted = model(image_corrupted)
s = encoder(mask_corrupted)  # descriptor degrades with corruption
```

**Multi-modal solution:**
```python
# Fusion provides complementary signals
s_mask = mask_encoder(mask_corrupted)      # geometric structure
s_img = image_encoder(image * mask)         # appearance/texture
s_fused = fusion(s_mask, s_img)            # robust combination

# Key: blur affects appearance ≠ noise affects both
# Fusion learns which modality to trust under which corruption
```

### **4-Week Execution Plan**

#### **Week 1: Implementation**

**Day 1-2: Extend Existing Encoder**
```python
# File: symalign/multimodal_encoder.py

import torch
import torch.nn as nn
import torchvision.models as models

class ImageRegionEncoder(nn.Module):
    """Extract appearance descriptor from masked image region."""
    def __init__(self, backbone='resnet18', descriptor_dim=32):
        super().__init__()
        # Use pretrained ResNet18 (lightweight)
        resnet = models.resnet18(pretrained=True)
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Small MLP to descriptor
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, descriptor_dim)
        )
        
    def forward(self, masked_image):
        """
        Args:
            masked_image: [B, 3, H, W] - image * mask
        Returns:
            descriptor: [B, descriptor_dim]
        """
        x = self.features(masked_image)  # [B, 512, 1, 1]
        descriptor = self.head(x)
        return F.normalize(descriptor, dim=1)


class MultiModalSymbolicEncoder(nn.Module):
    """Fuse mask-based and image-based descriptors."""
    def __init__(self, mask_encoder_path, descriptor_dim=32):
        super().__init__()
        
        # Load your existing mask encoder
        self.mask_encoder = self.load_mask_encoder(mask_encoder_path)
        # Freeze it initially (optional: fine-tune later)
        for param in self.mask_encoder.parameters():
            param.requires_grad = False
        
        # New: image encoder
        self.image_encoder = ImageRegionEncoder(
            backbone='resnet18',
            descriptor_dim=descriptor_dim
        )
        
        # Fusion module with attention
        self.fusion = CrossModalFusion(descriptor_dim=descriptor_dim)
        
    def forward(self, image, mask_soft):
        """
        Args:
            image: [B, 3, H, W] - original image
            mask_soft: [B, 1, H, W] - soft mask predictions
        Returns:
            s_fused: [B, descriptor_dim] - fused descriptor
            s_mask: [B, descriptor_dim] - mask-only (for ablations)
            s_img: [B, descriptor_dim] - image-only (for ablations)
        """
        # Mask-based descriptor (your existing encoder)
        s_mask = self.mask_encoder(mask_soft)
        
        # Image-based descriptor (new)
        # Mask out background to focus on foreground region
        masked_image = image * mask_soft  # [B, 3, H, W]
        s_img = self.image_encoder(masked_image)
        
        # Fusion
        s_fused = self.fusion(s_mask, s_img)
        
        return s_fused, s_mask, s_img
    
    def load_mask_encoder(self, path):
        # Load your existing trained encoder
        encoder = torch.load(path)
        encoder.eval()
        return encoder


class CrossModalFusion(nn.Module):
    """Attention-based fusion of two modalities."""
    def __init__(self, descriptor_dim=32):
        super().__init__()
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(descriptor_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # weights for [mask, image]
            nn.Softmax(dim=1)
        )
        # Fusion projection
        self.project = nn.Linear(descriptor_dim * 2, descriptor_dim)
        
    def forward(self, s_mask, s_img):
        """
        Args:
            s_mask: [B, D]
            s_img: [B, D]
        Returns:
            s_fused: [B, D]
        """
        # Concatenate
        concat = torch.cat([s_mask, s_img], dim=1)  # [B, 2D]
        
        # Compute attention weights
        weights = self.attention(concat)  # [B, 2]
        
        # Weighted combination
        w_mask = weights[:, 0:1]  # [B, 1]
        w_img = weights[:, 1:2]   # [B, 1]
        weighted = w_mask * s_mask + w_img * s_img
        
        # Project to final descriptor
        s_fused = self.project(concat)
        s_fused = F.normalize(s_fused, dim=1)
        
        return s_fused
```

**Day 3-4: Multi-Modal Contrastive Training**
```python
# File: symalign/multimodal_loss.py

class MultiModalContrastiveLoss(nn.Module):
    """
    Three contrastive objectives:
    1. Intra-modal consistency (mask aug1 vs mask aug2)
    2. Intra-modal consistency (image aug1 vs image aug2)
    3. Cross-modal alignment (mask vs image from same sample)
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def info_nce_loss(self, z1, z2):
        """Standard InfoNCE contrastive loss."""
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=z1.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    
    def forward(self, s_mask_aug1, s_mask_aug2, 
                s_img_aug1, s_img_aug2,
                s_fused_aug1, s_fused_aug2):
        """
        Args:
            s_mask_aug1/2: [B, D] - mask descriptors under two augs
            s_img_aug1/2: [B, D] - image descriptors under two augs
            s_fused_aug1/2: [B, D] - fused descriptors
        Returns:
            loss: scalar
        """
        # Intra-modal: mask consistency
        loss_mask = self.info_nce_loss(s_mask_aug1, s_mask_aug2)
        
        # Intra-modal: image consistency
        loss_img = self.info_nce_loss(s_img_aug1, s_img_aug2)
        
        # Cross-modal: mask and image should agree
        loss_cross = self.info_nce_loss(s_mask_aug1, s_img_aug1)
        
        # Fusion consistency
        loss_fused = self.info_nce_loss(s_fused_aug1, s_fused_aug2)
        
        # Combine
        total_loss = loss_mask + loss_img + 0.5 * loss_cross + loss_fused
        
        return total_loss, {
            'loss_mask': loss_mask.item(),
            'loss_img': loss_img.item(),
            'loss_cross': loss_cross.item(),
            'loss_fused': loss_fused.item()
        }
```

**Day 5-7: Training Script**
```python
# File: tools/train_multimodal_encoder.py

import torch
from symalign.multimodal_encoder import MultiModalSymbolicEncoder
from symalign.multimodal_loss import MultiModalContrastiveLoss

def train_multimodal_encoder(args):
    # Load source training data
    dataset = KvasirSourceDataset(
        root=args.dataset_root,
        split='train',
        return_images=True  # NEW: need images now
    )
    
    # Initialize multi-modal encoder
    encoder = MultiModalSymbolicEncoder(
        mask_encoder_path=args.mask_encoder_path,
        descriptor_dim=32
    ).to(args.device)
    
    # Loss and optimizer
    criterion = MultiModalContrastiveLoss(temperature=0.1)
    # Only train image encoder + fusion (mask encoder frozen)
    optimizer = torch.optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # Training loop
    for epoch in range(args.epochs):
        for batch in dataloader:
            images = batch['images'].to(args.device)      # [B, 3, H, W]
            masks_gt = batch['masks'].to(args.device)     # [B, 1, H, W]
            
            # Generate two augmented views
            aug1 = augment_both(images, masks_gt)  # returns (img_aug1, mask_aug1)
            aug2 = augment_both(images, masks_gt)  # returns (img_aug2, mask_aug2)
            
            # Forward pass on both views
            s_fused1, s_mask1, s_img1 = encoder(aug1['image'], aug1['mask'])
            s_fused2, s_mask2, s_img2 = encoder(aug2['image'], aug2['mask'])
            
            # Multi-modal contrastive loss
            loss, loss_dict = criterion(
                s_mask1, s_mask2,
                s_img1, s_img2,
                s_fused1, s_fused2
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: {loss_dict}")
    
    # Save encoder
    torch.save(encoder.state_dict(), args.output_path)
    return encoder


def augment_both(images, masks):
    """
    Apply SAME geometric augmentation to both image and mask.
    Different photometric augmentation to image only.
    """
    # Geometric (applied to both)
    angle = torch.rand(1) * 360
    images_rot = rotate(images, angle)
    masks_rot = rotate(masks, angle)
    
    # Photometric (applied to image only)
    images_photo = color_jitter(images_rot)
    
    return {'image': images_photo, 'mask': masks_rot}
```

#### **Week 2: Training and Validation**

**Goals:**
1. Train multi-modal encoder on source
2. Validate embeddings are meaningful
3. Verify fusion beats individual modalities

**Experiments:**
```bash
# Train encoder (12-24 hours on single GPU)
python tools/train_multimodal_encoder.py \
    --dataset_root ./segdata/kvasir \
    --mask_encoder_path ./runs/symalign_encoder_kvasir/encoder_final.pth \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_path ./runs/multimodal_encoder_kvasir/encoder_final.pth
```

**Validation Metrics:**
```python
# Compute retrieval accuracy
def validate_encoder(encoder, val_dataset):
    """
    For each sample, retrieve K nearest neighbors.
    Good encoder: neighbors have similar mask properties.
    """
    descriptors = []
    metadata = []
    
    for sample in val_dataset:
        img, mask = sample['image'], sample['mask']
        s_fused, s_mask, s_img = encoder(img, mask)
        
        descriptors.append(s_fused)
        metadata.append({
            'area': mask.sum().item(),
            'compactness': compute_compactness(mask),
            'sample_id': sample['id']
        })
    
    # Compute retrieval: are similar masks near each other?
    retrieval_acc = compute_knn_accuracy(descriptors, metadata, k=5)
    
    print(f"Retrieval accuracy (K=5): {retrieval_acc:.3f}")
    return retrieval_acc

# Ablation: which modality is most informative?
def ablate_modalities(encoder, val_dataset):
    """Compare mask-only, image-only, fused."""
    results = {
        'mask': evaluate_retrieval(mask_descriptors),
        'image': evaluate_retrieval(image_descriptors),
        'fused': evaluate_retrieval(fused_descriptors)
    }
    
    print(f"Retrieval accuracy:")
    print(f"  Mask-only:  {results['mask']:.3f}")
    print(f"  Image-only: {results['image']:.3f}")
    print(f"  Fused:      {results['fused']:.3f}")
    
    # Success criterion: fused > max(mask, image)
    return results
```

**Success Criteria (Week 2):**
- ✓ Retrieval accuracy > 0.6 (K=5)
- ✓ Fused beats both mask-only and image-only
- ✓ Embeddings cluster by structure (visualize with UMAP)

**Decision Point:**
- **If success criteria met:** Proceed to Week 3 (adaptation)
- **If not met:** Debug (likely issues: augmentation mismatch, fusion architecture)

#### **Week 3: Target Adaptation**

**Goal:** Test if multi-modal symbolic alignment beats baselines on mixed ops4 S4.

**Integration with existing adaptation code:**
```python
# Modify tools/adapt_baselines.py

def adapt_with_multimodal_symbolic(model, encoder, target_loader, args):
    """
    Adapt using multi-modal symbolic alignment.
    """
    # EMA priors (separate for each modality + fused)
    ema_mask = EMAStatistics(dim=32)
    ema_img = EMAStatistics(dim=32)
    ema_fused = EMAStatistics(dim=32)
    
    for step, batch in enumerate(target_loader):
        images = batch['images']  # [B, 3, H, W]
        
        # Forward pass
        preds = model(images)  # [B, 1, H, W] soft masks
        
        # Extract multi-modal descriptors
        with torch.no_grad():
            s_fused, s_mask, s_img = encoder(images, preds)
        
        # Update EMA priors (confidence-gated)
        confidence = compute_confidence(preds)
        if confidence > args.conf_thr:
            ema_mask.update(s_mask)
            ema_img.update(s_img)
            ema_fused.update(s_fused)
        
        # Symbolic alignment losses
        L_sym_mask = align_to_prior(s_mask, ema_mask)
        L_sym_img = align_to_prior(s_img, ema_img)
        L_sym_fused = align_to_prior(s_fused, ema_fused)
        
        # Total loss
        L_total = L_consistency(preds, augmented_preds) + \
                  args.teacher_kl_weight * L_kl(teacher, student) + \
                  args.lambda_sym * (L_sym_fused + 0.5 * L_sym_mask + 0.5 * L_sym_img)
        
        # Backward
        L_total.backward()
        optimizer.step()
```

**Experiments to run:**
```bash
# Baseline comparison (mixed ops4 S4)
for method in tent lora salt multimodal_sym; do
    python tools/adapt_baselines.py \
        --dataset_root ./segdata/kvasir \
        --adapt_manifest ./splits/kvasir_target_adapt.txt \
        --eval_manifest ./splits/kvasir_target_holdout.txt \
        --corruption mixed --severity 4 --num_ops 4 \
        --method $method \
        --multimodal_encoder_path ./runs/multimodal_encoder_kvasir/encoder_final.pth \
        --steps 500 --batch_size 4 --lr 1e-4 \
        --out_csv ./runs/multimodal_comparison_${method}.csv
done
```

**Expected Results (Week 3):**

| Method | Dice | IoU | BF | HD95 | Interpretation |
|--------|------|-----|-----|------|----------------|
| TENT | 0.7349 | 0.6152 | 0.2289 | 46.92 | Baseline |
| Your mask-only | 0.7376 | 0.6178 | 0.2274 | 46.36 | Current (marginal) |
| **Multi-modal (target)** | **0.760+** | **0.640+** | **0.250+** | **<44** | **Goal** |

**Success Criteria (Week 3):**
- **Minimum:** Beat TENT by 2+ Dice points (0.755+)
- **Ideal:** Beat TENT on Dice AND improve BF by 0.02+ (0.25+)

**Decision Point:**
- **If target met:** Proceed to ablations + write
- **If close (0.750-0.754):** Tune hyperparameters, still publishable at MICCAI
- **If fails (<0.745):** Fallback to Direction 3 (stratified)

#### **Week 4: Ablations and Paper**

**Critical Ablations:**

1. **Modality Ablation:**
```
Method              Dice    BF     Interpretation
------------------------------------------------
Mask-only          0.7376  0.2274  Current baseline
Image-only         ?.????  ?.????  New
Fused (ours)       0.760+  0.250+  Should be best
```

2. **Corruption-Specific Analysis:**
```python
# Which corruptions benefit most from fusion?
for corruption in ['blur', 'noise', 'jpeg', 'illumination']:
    results_mask = eval_on_corruption(mask_only, corruption, S4)
    results_fused = eval_on_corruption(fused, corruption, S4)
    
    improvement = results_fused - results_mask
    print(f"{corruption}: {improvement:.3f} Dice improvement")
```

**Hypothesis:** 
- Blur/JPEG (appearance-heavy): Large improvement
- Noise (affects both): Moderate improvement

3. **Attention Weight Analysis:**
```python
# Visualize what the fusion module learns
def analyze_fusion_weights(encoder, samples):
    """Show which modality gets higher weight under different corruptions."""
    for sample in samples:
        # Get attention weights
        weights = encoder.fusion.attention(concat)  # [2] for [mask, image]
        
        print(f"Sample {sample.id} (corruption: {sample.corruption}):")
        print(f"  Mask weight:  {weights[0]:.3f}")
        print(f"  Image weight: {weights[1]:.3f}")
```

**Expected:** Under blur, image weight decreases; under noise, balanced.

4. **PEFT Compatibility:**
```
Method                          Dice    BF
-------------------------------------------
PEFT-only (LoRA)               0.7418  0.2375
Mask-symbolic + LoRA           0.7317  0.2285  (worse!)
Multi-modal + LoRA             0.770+  0.255+  (better!)
```

**Paper Outline:**
```
Title: "Multi-Modal Symbolic Alignment for Robust Source-Free Segmentation Adaptation"

Abstract:
- Problem: source-free adaptation under severe appearance shift
- Limitation: mask-only descriptors degrade under corruption
- Solution: fuse mask structure + image appearance descriptors
- Results: X% improvement over SOTA on Kvasir under severe shift

1. Introduction
   - Source-free adaptation motivation
   - Current limitations (your failed mask-only results become motivation!)
   - Multi-modal fusion as solution

2. Related Work
   - Source-free adaptation
   - Contrastive learning for medical imaging
   - Multi-modal self-supervised learning

3. Method
   - Multi-modal encoder architecture
   - Cross-modal contrastive training
   - Symbolic alignment during adaptation
   - Fusion mechanism

4. Experiments
   - Setup: Kvasir, mixed ops4 stress regime
   - Baselines: TENT, PEFT-only, mask-only symbolic
   - Main results: multi-modal beats all
   - Ablations: modality comparison, corruption analysis

5. Analysis
   - Attention weight visualization
   - Failure case analysis
   - Corruption-specific benefits

6. Conclusion
   - Multi-modal fusion is key for robustness
   - Future: extend to other medical imaging tasks
```

### **Risk Mitigation**

**Risk 1: Multi-modal doesn't beat mask-only**
- **Mitigation:** Start with good image encoder (pretrained ResNet18)
- **Fallback:** If marginal gain, reframe as "understanding modal contributions"

**Risk 2: Fusion module doesn't learn meaningful weights**
- **Mitigation:** Start with simple attention, add complexity if needed
- **Debug:** Log attention weights during training

**Risk 3: Image encoder too expensive**
- **Mitigation:** Use ResNet18 (lightweight, 11M params)
- **Alternative:** MobileNetV2 if needed

### **Deliverables (End of Week 4)**

1. **Code:**
   - `symalign/multimodal_encoder.py`
   - `symalign/multimodal_loss.py`
   - `tools/train_multimodal_encoder.py`
   - Updated `tools/adapt_baselines.py`

2. **Results:**
   - Comparison table (baseline vs multi-modal)
   - Ablation tables (modality comparison)
   - Corruption analysis figure
   - Attention weight visualization

3. **Paper Draft:**
   - 6-page main paper (CVPR format)
   - 2-page appendix (additional ablations)

---

## **DIRECTION 3: Stratified Contrastive Learning** (FALLBACK)

### **Core Innovation**
Current encoder likely learns "big vs small polyps" instead of structural properties. Stratify training by size/complexity to force encoder to learn meaningful structure.

### **Why This is Simpler**

**Minimal changes to existing code:**
```python
# Current training (what you have)
loss = contrastive_loss(descriptors_batch)

# Stratified version (small change)
def stratified_contrastive_loss(masks_batch, descriptors_batch):
    # Compute mask statistics
    areas = masks_batch.sum(dim=[1,2,3])
    
    # Stratify into 3 groups
    small_idx = areas < area_quantile_33
    medium_idx = (areas >= area_quantile_33) & (areas < area_quantile_66)
    large_idx = areas >= area_quantile_66
    
    # Separate contrastive loss per stratum
    loss_small = contrastive_loss(descriptors_batch[small_idx])
    loss_medium = contrastive_loss(descriptors_batch[medium_idx])
    loss_large = contrastive_loss(descriptors_batch[large_idx])
    
    # Weighted combination (inversely weighted by stratum size)
    loss = loss_small + loss_medium + loss_large
    return loss
```

### **2-Week Execution Plan**

**Week 1: Implementation + Retraining**
- Day 1-2: Implement stratified sampling
- Day 3-5: Retrain encoder on source
- Day 6-7: Validate embeddings

**Week 2: Adaptation + Results**
- Day 1-3: Run adaptation experiments
- Day 4-7: Write paper (target MICCAI)

### **Success Criteria**
- Stratified encoder beats current encoder by 1+ Dice point
- Retrieval accuracy improves on small/complex polyps

### **Venue Target**
- **Primary:** MICCAI (70% chance)
- **Backup:** MIDL workshop (90% chance)

---

## **DIRECTION 2: Equivariant Descriptors** (IF TIME PERMITS)

### **Core Innovation**
Make encoder SE(2)-equivariant so rotation/scale changes to mask produce predictable descriptor transformations.

### **Why This is Harder**

Requires new library and careful design:
```python
from escnn import nn as enn

class EquivariantEncoder(enn.EquivariantModule):
    def __init__(self, in_type, out_type):
        super().__init__()
        self.conv1 = enn.R2Conv(in_type, 16, kernel_size=3)
        self.conv2 = enn.R2Conv(16, 32, kernel_size=3)
        # ... SE(2) group operations throughout
```

### **6-Week Execution Plan**

**Weeks 1-2: Learn escnn library**
- Study examples, implement toy models
- Understand group theory basics

**Weeks 3-4: Implement equivariant encoder**
- Replace your encoder with equivariant version
- Debug group operations

**Weeks 5-6: Train and evaluate**
- Expect similar performance but with theoretical guarantees

### **Success Criteria**
- Equivariant encoder matches standard encoder performance
- Requires less augmentation (built-in invariance)
- Can prove theoretical properties

### **Venue Target**
- **Primary:** CVPR/ICCV (theory + empirical = strong paper)
- **Timeline:** 6 weeks minimum

---

## **DIRECTION 1: Topological Descriptors** (AMBITIOUS)

### **Core Innovation**
Use persistent homology to capture topological features (holes, connected components) that survive under corruption.

### **Why This is Most Ambitious**

Complex implementation:
```python
from giottotda.persistent_homology import VietorisRipsPersistence
from giottotda.diagrams import PersistenceEntropy

class TopologicalEncoder(nn.Module):
    def __init__(self):
        self.persistence = VietorisRipsPersistence(homology_dimensions=[0, 1])
        self.vectorizer = PersistenceDiagramVectorizer()
        
    def forward(self, mask):
        # Compute persistence diagram
        dgm = self.persistence.fit_transform(mask)
        # Vectorize (non-differentiable!)
        vector = self.vectorizer(dgm)
        return vector
```

**Challenges:**
1. Persistence computation is slow
2. Making it differentiable is non-trivial
3. High learning curve

### **8-Week Execution Plan**

**Weeks 1-3: Learn topology + libraries**
- Understand persistent homology
- Study giotto-tda, pytorch-topological
- Implement toy examples

**Weeks 4-6: Implement topological encoder**
- Integrate with your pipeline
- Make differentiable (use approximations)

**Weeks 7-8: Train and evaluate**
- Expect unique structural insights

### **Success Criteria**
- Topological features capture fragmentation, holes
- Robust to noise (topology is stable under perturbations)
- Novel visualizations of what topology captures

### **Venue Target**
- **Primary:** NeurIPS (highest novelty, theory + application)
- **Timeline:** 8 weeks minimum
- **Risk:** High - may not work at all

---

## **RECOMMENDED EXECUTION STRATEGY**

### **Primary Path: Multi-Modal (4 weeks)**
```
Week 1: Implement multi-modal encoder
Week 2: Train and validate
Week 3: Adapt on target
Week 4: Ablations + write

Decision point (end Week 3):
├─ Success (Dice 0.760+) → Target CVPR/ICCV (add 2 datasets)
├─ Moderate (Dice 0.750-0.759) → Target MICCAI (polish)
└─ Failure (Dice <0.750) → Execute fallback
```

### **Fallback Path: Stratified (2 weeks)**
```
If multi-modal fails by end Week 3:
Week 4: Implement stratified contrastive
Week 5: Retrain encoder
Week 6: Run experiments + write for MICCAI
```

### **Contingency: If Both Fail**
```
Write up as negative result:
"On the Limits of Learned Symbolic Alignment for Source-Free Adaptation"
- Show what doesn't work and why
- Target workshop (CVPR, ICCV, MICCAI workshops)
- Still publishable, useful for community
```

---

## **Final Recommendation**

**START IMMEDIATELY with Direction 4 (Multi-Modal).** Here's your to-do list for tomorrow:

1. **Morning:** Implement `ImageRegionEncoder` class
2. **Afternoon:** Implement `MultiModalSymbolicEncoder` wrapper
3. **Evening:** Start implementing `MultiModalContrastiveLoss`

By end of Week 1, you should have trained encoder. By end of Week 3, you know if you have a CVPR paper.

**Don't implement all 4 directions.** Commit to multi-modal. If it works, you're done. If it fails, pivot to stratified (2 more weeks).

