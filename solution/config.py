"""
Hyperparameter configuration for the Floorplan Transformer.
All modules should import settings from here.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent          # solution/
REPO_ROOT    = PROJECT_ROOT.parent            # FloorSet/
CONTEST_DIR  = REPO_ROOT / "iccad2026contest"

LOGS_DIR        = PROJECT_ROOT / "logs"
VIZ_DIR         = LOGS_DIR / "viz"
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
CACHE_DIR       = PROJECT_ROOT / "cache"

# Number of preprocessed samples stored in each shard file
SHARD_SIZE = 2000

# Load entire cache into RAM at training startup (requires ~50 GB for 1 M samples).
# Set True when RAM is sufficient; eliminates all disk I/O during training.
CACHE_PRELOAD = True

# =============================================================================
# Data
# =============================================================================

# Canvas normalization factor (1.2 leaves ~30% routing headroom)
CANVAS_K = 1.2

# Padding sentinel value used by FloorSet loaders
PAD_VALUE = -1.0

# Largest possible block count in any sample
MAX_BLOCKS = 120

# boundary_type bitmask → 8-dim one-hot index mapping
# constraints[:, 4] stores one of: 0, 1, 2, 4, 5, 6, 8, 9, 10
BOUNDARY_CODE_TO_IDX = {
    1:  0,   # Left
    2:  1,   # Right
    4:  2,   # Top
    8:  3,   # Bottom
    5:  4,   # Top-left
    6:  5,   # Top-right
    9:  6,   # Bottom-left
    10: 7,   # Bottom-right
}
BOUNDARY_DIM = 8  # one-hot dimensionality (0 → all-zero vector, no constraint)

# Cluster group IDs in the dataset: 1, 2, 3, 4  (0 = no cluster)
CLUSTER_DIM = 4   # one-hot dimensionality for cluster group

# Raw token feature dimensionality before Linear projection:
#   area_target(1) + block_type(3) + target_w/h(2) + target_x/y(2)
#   + boundary_type(8) + is_mib(1) + cluster_group_onehot(4) = 21
RAW_FEATURE_DIM = 21

# =============================================================================
# Model Architecture
# =============================================================================
GRID_SIZE         = 64   # discrete placement grid (G × G)
D_MODEL           = 256
NUM_HEADS         = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD   = 1024
DROPOUT           = 0.1

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE    = 64
MAX_EPOCHS    = 100
PATIENCE      = 10          # early stopping patience (epochs)

LEARNING_RATE  = 1e-4
WARMUP_STEPS   = 4000       # linear warmup before cosine decay
GRAD_CLIP_NORM = 1.0

# Loss weights
LAMBDA_WIRELENGTH = 0.3     # weight for L_wirelength
LAMBDA_AREA       = 0.3     # weight for L_area
LAMBDA_VIOLATION  = 0.4     # weight for L_violation

# Extra weight on preplaced blocks in L_coord
PREPLACED_COORD_WEIGHT = 5.0

# =============================================================================
# Evaluation / Validation
# =============================================================================

# Validate every N epochs (runs official evaluator)
VALIDATE_EVERY = 5

# Block sizes selected for visualization (one per 10 sizes from 21 to 111)
VIZ_BLOCK_SIZES = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111]
