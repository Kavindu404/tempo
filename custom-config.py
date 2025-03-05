import torch

class CFG:
    IMG_PATH = './dataset'
    ANNOTATIONS_PATH = './dataset/annotations/annotations.json'
    TRAIN_IMAGES_DIR = './dataset/images/train'
    VAL_IMAGES_DIR = './dataset/images/val'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset name
    DATASET = "custom_dataset"
    
    TRAIN_DDP = False  # Set this to False for single GPU training
    NUM_WORKERS = 2
    PIN_MEMORY = True
    LOAD_MODEL = False

    # You might need to adjust this based on your dataset
    N_VERTICES = 192  # maximum number of vertices per image in dataset
    
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

    BATCH_SIZE = 16  # Adjust based on your GPU memory
    START_EPOCH = 0
    NUM_EPOCHS = 100  # You can adjust this
    MILESTONE = 0
    SAVE_BEST = True
    SAVE_LATEST = True
    SAVE_EVERY = 10
    VAL_EVERY = 1

    MODEL_NAME = f'vit_small_patch{PATCH_SIZE}_{INPUT_SIZE}_dino'
    NUM_PATCHES = int((INPUT_SIZE // PATCH_SIZE) ** 2)

    LR = 4e-4
    WEIGHT_DECAY = 1e-4

    generation_steps = (N_VERTICES * 2) + 1
    run_eval = False

    EXPERIMENT_NAME = f"train_Pix2Poly_{DATASET}_singleGPU_{MODEL_NAME}_initialLR_{LR}_bs_{BATCH_SIZE}_Nv_{N_VERTICES}_Nbins{NUM_BINS}_{NUM_EPOCHS}epochs"

    if "debug" in EXPERIMENT_NAME:
        BATCH_SIZE = 10
        NUM_WORKERS = 0
        SAVE_BEST = False
        SAVE_LATEST = False
        SAVE_EVERY = NUM_EPOCHS
        VAL_EVERY = 50

    if LOAD_MODEL:
        CHECKPOINT_PATH = f"runs/{EXPERIMENT_NAME}/logs/checkpoints/latest.pth"
    else:
        CHECKPOINT_PATH = ""