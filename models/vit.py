import torch
import torch.nn as nn

from config import NUM_CHANNELS, EMBED_DIM, PATCH_SIZE, ATTENTION_HEADS,\
      MLP_HIDDEN_LAYER_NODES, NUM_CLASSES, PATCHES, TRANSFORMER_BLOCKS


"""
Part 1 of Vision Transformer architecture - PatchEmbedding
"""
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(NUM_CHANNELS, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE) # non-overlapping patches

    def forward(self, x):
        # batch_size = 64, patches = 16 (in each image), embed_dim = 20 (each image) -> (64, 16, 20)
        # so after conv2d we will get [64, embed_dim, 4, 4] (4 = num of patches along x , y dim)
        # we need to convert it
        x = self.patch_embed(x) # perform patch embedding torch.Size([64, 20, 4, 4])
        x = x.flatten(2) # flatten to preserve only 3 dims torch.Size([64, 20, 16])
        x = x.transpose(1, 2) # 0th dim untouched. Swap 1st and 2nd dim. torch.Size([64, 16, 20])
        return x


"""
Part 2 - Transformer encoder
    - Layer Normalization
    - Multi Head attn
    - Layer Normalization
    - Residuals
    - MLP
        - Activation function
"""
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Normalization along embed dim
        self.layer_norm1 = nn.LayerNorm(EMBED_DIM)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=EMBED_DIM, num_heads=ATTENTION_HEADS, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=EMBED_DIM, out_features=MLP_HIDDEN_LAYER_NODES),
            nn.GELU(),
            nn.Linear(in_features=MLP_HIDDEN_LAYER_NODES, out_features=EMBED_DIM),
        )

    def forward(self, x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.multi_head_attn(x, x, x)[0] # input: pass (Q, K, V) output: get first location
        x = x + residual1 # multi-head attn + residual
        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual2
        return x


"""
Part 3 - MLP Head for classification
"""
class MLP_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(EMBED_DIM)
        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=EMBED_DIM, out_features=NUM_CLASSES),
        )

    def forward(self, x):
        # x = x[:,0] # CLS token
        x = self.layer_norm(x)
        x = self.mlp_head(x)
        return x


"""
Final Part: Put it all together
"""
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        # [Learnable parameters] We need to add CLS token, positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM)) # initialize with random tensor. No. of CLS tokens needed = no. of images in our batch. # dim of CLS = EMBED_DIM
        self.position_embedding = nn.Parameter(torch.randn(1, PATCHES + 1, EMBED_DIM)) # initialize with random tensor. No. of pos embedding needed = no. of patches + 1(CLS token)
        transformer_blocks = []
        for _ in range(TRANSFORMER_BLOCKS):
            transformer_blocks.append(TransformerEncoder())
        self.transformer_blocks = nn.Sequential(*transformer_blocks) # unpack the list since nn.Sequential expects Module instead of List.
        self.mlp_head = MLP_Head()

    def forward(self, x):
        x = self.patch_embedding(x)
        # append CLS token
        B = x.shape[0] # 0th dimension is the batch size. Do not use `BATCH_SIZE` as total num of images is 
                       # not exactly divisible by `BATCH_SIZE`. Then in the last batch there might be lesser 
                       # images. So we cannot hardcode it. 
        cls_tokens = self.cls_token.expand(B, -1, -1) # expand along the batch dimension. Now there are `B` tokens.
        x = torch.cat((cls_tokens, x), dim=1) # concat cls_tokens + input images (append)
        # add positional embedding
        x = x + self.position_embedding
        x = self.transformer_blocks(x)
        # we need to pass only CLS token to the MLP head for classification
        x = x[:,0] # CLS token
        x = self.mlp_head(x)
        return x
