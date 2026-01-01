# define variables
BATCH_SIZE = 64
NUM_CLASSES = 10
IMG_SIZE = 28 # mnist 28 X 28
NUM_CHANNELS = 1 # black-white only images
PATCH_SIZE = 7 # image partition 4 along height and row. 
PATCHES = (IMG_SIZE // PATCH_SIZE) * (IMG_SIZE // PATCH_SIZE) # Total patches = 4*4 = 16
ATTENTION_HEADS = 4 # Multi-head attn (stacked in parallel)
EMBED_DIM = 20 # size of each vector after we convert patch to vector representation
TRANSFORMER_BLOCKS = 4 # how many times will the transformer encoder block will repeat (in series).
MLP_HIDDEN_LAYER_NODES = 64 # project to higher dimension and project back. 
LEARNING_RATE = 0.001
EPOCHS = 5