import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embedding_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Shape: (batch_size, embedding_dim, h_patches, w_patches)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embedding_dim)
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, attn_dropout):
        super(MultiheadSelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=attn_dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        return x

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_size, dropout):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x_residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.norm(x + x_residual)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_size, attn_dropout, mlp_dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attention = MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
        self.mlp = MLPBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.mlp(x)
        return x

class ViT(nn.Module):
    def __init__(
        self, img_size=224, in_channels=3, patch_size=16, num_transformer_layers=12, 
        embedding_dim=768, mlp_size=3072, num_heads=12, attn_dropout=0.1, 
        mlp_dropout=0.1, embedding_dropout=0.1, num_classes=1000
    ):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embedding_dim)
        self.num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, attn_dropout, mlp_dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        
        class_token = self.class_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embedding_dim)
        x = torch.cat((class_token, x), dim=1)  # Add class token to the patches
        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = x.permute(1, 0, 2)  # Required shape for nn.MultiheadAttention: (seq_len, batch_size, embedding_dim)
        x = self.encoder(x)
        x = x[0]  # Take the class token output
        x = self.classifier(x)

        return x
