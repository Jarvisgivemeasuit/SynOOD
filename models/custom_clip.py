import torch
from torch import nn
import clip

class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        feat_dim = 512
        self.proj = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, feat_dim),
        torch.nn.LayerNorm(feat_dim),
        torch.nn.ReLU(),
        ).to('cuda')

    def forward(self, image, prompts, tokenized_prompt):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features  = self.text_encoder(prompts, tokenized_prompt)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
    def image_forward(self, image, prompts_embeddings):
        image_features =  self.image_encoder(image.type(self.dtype)).float()
        # image_features = image_features + self.proj(image_features).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = prompts_embeddings / prompts_embeddings.norm(dim=-1, keepdim=True).float()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x