import torch.nn as nn
from transformers import logging
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPProcessor,
    CLIPModel,
)
import torch

logging.set_verbosity_error()


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = None

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        additional_tokens=[
            "pororo",
            "loopy",
            "eddy",
            "harry",
            "poby",
            "tongtong",
            "crong",
            "rody",
            "petty",
        ],
        max_length=85,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        self.device = device
        self.add_tokens(additional_tokens)
        # self.freeze()

    def add_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        # resize_position_embeddings
        old_embeddings = self.transformer.text_model.embeddings.position_embedding
        new_embeddings = self.transformer._get_resized_embeddings(old_embeddings, self.max_length)
        self.transformer.text_model.embeddings.position_embedding = new_embeddings
        self.transformer.config.max_position_embeddings = self.max_length
        self.transformer.max_position_embeddings = self.max_length
        self.transformer.text_model.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs_txt = self.transformer(input_ids=tokens)

        z = outputs_txt.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class CLIPVisionEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for images (from huggingface)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        self.transformer = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.device = device
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        z = self.transformer.get_image_features(**inputs)
        return z

    def encode(self, images):
        return self(images)

class CLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text and images (from huggingface)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        additional_tokens=[
            "pororo",
            "loopy",
            "eddy",
            "harry",
            "poby",
            "tongtong",
            "crong",
            "rody",
            "petty",
        ],
        max_length=85,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPModel.from_pretrained(version)
        self.add_tokens(additional_tokens)
        # self.vision_transformer = CLIPVisionModel.from_pretrained(version)
        # self.processor = CLIPProcessor.from_pretrained(version)
        self.device = device
        self.max_length = max_length

    def add_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens)
        self.transformer.resize_token_embeddings(len(self.tokenizer))

    def forward(self, text=None, images=None):
        z1, z2 = None, None

        if text is not None:
            batch_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(self.device)
            outputs_txt = self.transformer(input_ids=tokens)
            z1 = outputs_txt.last_hidden_state

        if images is not None:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            z2 = self.transformer.get_image_features(**inputs)
            if text is None:
                return z2
        else:
            return z1
        
        return z1, z2
    
    def encode(self, text=None, images=None):
        if text is not None:
            c = self(text=text)
        elif images is not None:
            c = self(images=images)
        return c