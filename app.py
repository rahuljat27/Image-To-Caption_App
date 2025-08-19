import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.distributions import Categorical
from transformers import AutoTokenizer
from PIL import Image
import io
import math

# --- Model Hyperparameters (from the provided notebook) ---
image_size = 128
hidden_size = 192
num_layers = (6, 6)
num_heads = 8
patch_size = 8
sos_token_id = 101  # [CLS]
eos_token_id = 102  # [SEP]

# --- Model Architecture Classes (from the provided notebook) ---
def extract_patches(image_tensor, patch_size=16):
    bs, c, h, w = image_tensor.size()
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    return unfolded

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, masking=True):
        super(AttentionBlock, self).__init__()
        self.masking = masking
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )

    def forward(self, x_in, kv_in, key_mask=None, key_padding_mask=None):
        if self.masking:
            bs, l, h = x_in.shape
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        else:
            mask = None
        
        return self.multihead_attn(
            x_in, 
            kv_in, 
            kv_in, 
            attn_mask=mask, 
            key_padding_mask=key_padding_mask
        )[0]

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True):
        super(TransformerBlock, self).__init__()
        self.decoder = decoder
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, masking=masking)
        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, masking=False)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
        # Corrected line: pass input_key_mask as key_padding_mask
        x = self.attn1(x, x, key_padding_mask=input_key_mask) + x
        x = self.norm1(x)
        if self.decoder:
            # Fix for the cross-attention call
            x = self.attn2(x, kv_cross, key_padding_mask=cross_key_mask) + x
            x = self.norm2(x)
        x = self.mlp(x) + x
        return self.norm_mlp(x)

class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.embedding.weight.data = 0.001 * self.embedding.weight.data
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, decoder=True) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, num_emb)
    def forward(self, input_seq, encoder_output, input_padding_mask=None, encoder_padding_mask=None):
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
        for block in self.blocks:
            embs = block(embs, input_key_mask=input_padding_mask, cross_key_mask=encoder_padding_mask, kv_cross=encoder_output)
        return self.fc_out(embs)

class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, num_layers=3, num_heads=4):
        super(VisionEncoder, self).__init__()
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, decoder=False, masking=False) for _ in range(num_layers)])
    def forward(self, image):
        bs = image.shape[0]
        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)
        embs = patch_emb + self.pos_embedding
        for block in self.blocks:
            embs = block(embs)
        return embs

class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16, hidden_size=128, num_layers=(3, 3), num_heads=4):
        super(VisionEncoderDecoder, self).__init__()
        self.encoder = VisionEncoder(
            image_size=image_size,
            channels_in=channels_in,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers[0],
            num_heads=num_heads
        )
        self.decoder = Decoder(
            num_emb=num_emb,
            hidden_size=hidden_size,
            num_layers=num_layers[1],
            num_heads=num_heads
        )
    def forward(self, input_image, target_seq, padding_mask):
        bool_padding_mask = padding_mask == 0
        encoded_seq = self.encoder(image=input_image)
        decoded_seq = self.decoder(
            input_seq=target_seq,
            encoder_output=encoded_seq,
            input_padding_mask=bool_padding_mask
        )
        return decoded_seq

# --- Caption Generation Logic (from the provided notebook) ---
def generate_caption(model, image, tokenizer, device, max_len=50, temp=0.5):
    model.eval()
    with torch.no_grad():
        image_embedding = model.encoder(image.to(device))
        log_tokens = [torch.tensor([[sos_token_id]], device=device)]
        for _ in range(max_len):
            input_tokens = torch.cat(log_tokens, dim=1).to(device)
            data_pred = model.decoder(input_tokens, image_embedding)
            dist = Categorical(logits=data_pred[:, -1] / temp)
            next_token = dist.sample().reshape(1, 1)
            log_tokens.append(next_token)
            if next_token.item() == eos_token_id:
                break
        pred_tensor = torch.cat(log_tokens, dim=1).cpu()
        pred_text = tokenizer.decode(pred_tensor[0].tolist(), skip_special_tokens=True)
        return pred_text

# --- Streamlit Application ---
st.title("Image Captioning with Vision Transformer & Decoder")

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    caption_model = VisionEncoderDecoder(
        image_size=image_size,
        channels_in=3,
        num_emb=tokenizer.vocab_size,
        patch_size=patch_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads
    ).to(device)
    
    # Load the best-performing model weights
    try:
        # Assumes the checkpoint is in the same directory
        checkpoint = torch.load("caption_model_epoch29.pth", map_location=device)
        caption_model.load_state_dict(checkpoint)
        caption_model.eval()
    except FileNotFoundError:
        st.error("Model checkpoint `caption_model_epoch29.pth` not found. Please upload it or ensure it's in the same directory as the script.")
        return None, None, None
    
    return caption_model, tokenizer, device

caption_model, tokenizer, device = load_model_and_tokenizer()

if caption_model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for the model
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)

        if st.button("Generate Caption"):
            st.write("Generating caption...")
            caption = generate_caption(caption_model, image_tensor, tokenizer, device)
            st.success(f"Generated Caption: {caption}")