import torch

class CFG:
    # Path settings
    DATASET = "custom_dataset"
    TRAIN_DATASET_DIR = "./dataset"
    VAL_DATASET_DIR = "./dataset"
    TEST_IMAGES_DIR = "./dataset/images/val"
    TRAIN_ANNOTATIONS_FILE = "./dataset/annotations/640.json"
    VAL_ANNOTATIONS_FILE = "./dataset/annotations/640.json"
    
    # Model settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_VERTICES = 192  # maximum number of vertices per image in dataset.
    SINKHORN_ITERATIONS = 100
    MAX_LEN = (N_VERTICES*2) + 2
    IMG_SIZE = 224
    INPUT_SIZE = 224
    PATCH_SIZE = 8
    INPUT_HEIGHT = INPUT_SIZE
    INPUT_WIDTH = INPUT_SIZE
    NUM_BINS = INPUT_HEIGHT*1
    LABEL_SMOOTHING = 0.0
    vertex_loss_weight = 1.0
    perm_loss_weight = 10.0
    SHUFFLE_TOKENS = False
    
    # Training settings
    BATCH_SIZE = 8  # Adjust based on your GPU memory
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False
    START_EPOCH = 0
    NUM_EPOCHS = 100
    MILESTONE = 0
    SAVE_BEST = True
    SAVE_LATEST = True
    SAVE_EVERY = 10
    VAL_EVERY = 1

    # Model architecture
    MODEL_NAME = f'vit_small_patch{PATCH_SIZE}_{INPUT_SIZE}_dino'
    NUM_PATCHES = int((INPUT_SIZE // PATCH_SIZE) ** 2)

    # Optimizer settings
    LR = 4e-4
    WEIGHT_DECAY = 1e-4

    # Generation settings
    generation_steps = (N_VERTICES * 2) + 1  # sequence length during prediction
    run_eval = False

    # Experiment name
    EXPERIMENT_NAME = f"train_Pix2Poly_custom_dataset_run1_{MODEL_NAME}_Linear_{vertex_loss_weight}xVertexLoss_{perm_loss_weight}xPermLoss_bs_{BATCH_SIZE}_Nv_{N_VERTICES}_Nbins{NUM_BINS}_{NUM_EPOCHS}epochs"

    # Checkpoint path (if loading a model)
    if LOAD_MODEL:
        CHECKPOINT_PATH = f"runs/{EXPERIMENT_NAME}/logs/checkpoints/latest.pth"  # full path to checkpoint to be loaded if LOAD_MODEL=True
    else:
        CHECKPOINT_PATH = ""
