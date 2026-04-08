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
CACHE_PRELOAD = False

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
D_MODEL           = 256
NUM_HEADS         = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD   = 1024
DROPOUT           = 0.1

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE    = 128
MAX_EPOCHS    = 100
PATIENCE      = 10          # early stopping patience (epochs)

LEARNING_RATE  = 1e-4
WARMUP_STEPS   = 4000       # linear warmup before cosine decay
GRAD_CLIP_NORM = 1.0

# Loss weights
LAMBDA_WIRELENGTH = 0.3     # weight for L_wirelength
LAMBDA_AREA       = 0.3     # weight for L_area
LAMBDA_GROUPING   = 0.1     # weight for V_grouping  (cluster centroid distance)
LAMBDA_MIB        = 0.1     # weight for V_mib       (macro-in-block size deviation)
LAMBDA_BOUNDARY   = 0.1     # weight for V_boundary  (boundary gap penalty)
LAMBDA_OVERLAP    = 50.0    # weight for V_overlap   (pairwise overlap area)
LAMBDA_COORD      = 1.0     # weight for L_coord     (L1 position loss)

# Ratio regularisation: penalise log(w/h)² to prevent degenerate thin-strip blocks.
# Weight decays linearly from LAMBDA_RATIO_REG → 0 over RATIO_REG_DECAY_EPOCHS epochs,
# then stays at 0 so physical losses fully take over.
LAMBDA_RATIO_REG        = 2.0   # initial weight (epoch 0)
RATIO_REG_DECAY_EPOCHS  = 20    # number of epochs to reach 0

# Aspect ratio (w/h) hard clamp in ContinuousRegressionHead
MIN_RATIO = 0.2   # w/h lower bound (block at most 5× taller than wide)
MAX_RATIO = 5.0   # w/h upper bound (block at most 5× wider than tall)


# =============================================================================
# Evaluation / Validation
# =============================================================================

# Validate every N epochs (runs official evaluator)
VALIDATE_EVERY = 5

# Block sizes selected for visualization (one per 10 sizes from 21 to 111)
VIZ_BLOCK_SIZES = [21, 31, 41, 51, 61, 71, 81, 91, 101, 111]
