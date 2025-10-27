import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm

# ======================================
# TRANSFORMER COMPONENTS
# ======================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_padding_mask(self, seq):
        return (seq != 0).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask

    def encode(self, src):
        src_mask = self.create_padding_mask(src)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask=None):
        tgt_mask = self.create_padding_mask(tgt)
        look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        combined_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, combined_mask)
        return self.output_layer(x)

    def forward(self, src, tgt):
        src_mask = self.create_padding_mask(src)
        enc_output = self.encode(src)
        output = self.decode(tgt[:, :-1], enc_output, src_mask)
        return output

# ======================================
# DECODING (INFERENCE)
# ======================================

def greedy_decode(model, src, max_len, device, bos_id=2, eos_id=3):
    model.eval()
    src = src.to(device)
    with torch.no_grad():
        enc_output = model.encode(src)
        src_mask = model.create_padding_mask(src)
        tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for _ in range(max_len):
            output = model.decode(tgt, enc_output, src_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == eos_id:
                break
        return tgt[0].cpu().tolist()

# ======================================
# STREAMLIT UI
# ======================================

st.title("🗣️ Urdu Conversational Chatbot")

model_file = st.file_uploader("📦 Upload your trained model (.pth)", type=["pth"])
sp_model_file = st.file_uploader("📜 Upload your SentencePiece model (.model)", type=["model"])

if model_file and sp_model_file:
    with open("uploaded_model.pth", "wb") as f:
        f.write(model_file.read())
    with open("uploaded_sp.model", "wb") as f:
        f.write(sp_model_file.read())

    st.success("✅ Files uploaded successfully!")

    sp = spm.SentencePieceProcessor()
    sp.load("uploaded_sp.model")

    vocab_size = len(sp)
    model = TransformerSeq2Seq(vocab_size)
    checkpoint = torch.load("uploaded_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    user_input = st.text_input("💬 Type your message in Urdu:")

    if user_input:
        src_ids = [2] + sp.encode(user_input, out_type=int) + [3]
        src_tensor = torch.tensor([src_ids], dtype=torch.long)
        output_ids = greedy_decode(model, src_tensor, max_len=64, device='cpu')
        response = sp.decode([id for id in output_ids if id not in [0, 2, 3]])
        st.write("🤖 **Bot:**", response)
