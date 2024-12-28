import os
import json
import streamlit as st
from nltk.tree import Tree
from nltk.tree.prettyprinter import TreePrettyPrinter
import numpy as np
import pandas as pd
import torch
from torch import nn

# ----------------------
# Model Information
# ----------------------
model_info = {
    "s_transformer-e": {
        "subheader": "Model: Multi-Head Transformer Encoder",
        "pre_processing": """
Dataset = Penn TreeBank Dataset
Embedding Model = GloVe("6B.200d")
        """,
        "parameters": """
Batch Size = 32
Embedding Size = 200
Number of Attention Heads = 8
Number of Encoder Layers = 2
Feedforward Hidden Size = 512
Dropout Rate = 0.16426146772147993
Learning Rate = 0.001102590574546097
Epochs = 20
Optimizer = AdamW
Weight Decay = 0.01
Loss Function = CrossEntropyLoss
Hyperparameter Tuning: Bayesian Optimization
        """,
        "model_code": """
class Encoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(Encoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_dim, padding_idx, embedding_matrix, dropout=0.1):
        super(Model, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = True
        # Positional Encoding Layer
        self.pos_encoder = SPositionalEncoding(embed_size)
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_size))
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        # Fully Connected Layer
        self.fc = nn.Linear(embed_size, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        src_key_padding_mask = (x == self.embedding.padding_idx)
        transformer_output = self.transformer_encoder(
            embedded, 
            src_key_padding_mask=src_key_padding_mask
        )
        transformer_output = self.dropout(transformer_output)
        logits = self.fc(transformer_output)
        return logits
        """,
        "forward_pass": {
        "Embedding": r'''
        \mathbf{E} = \text{Embedding}(x) \\~~\\
        \mathbf{E} \in \mathbb{R}^{T \times d}
        ''',
        "Positional Encoding": r'''
        \mathbf{E}' = \text{PositionalEncoding}(\mathbf{E}) \\~~\\
        \mathbf{E}' \in \mathbb{R}^{T \times d}
        ''',
        "Dropout (Pre-Transformer)": r'''
        \tilde{\mathbf{E}} = \text{Dropout}(\mathbf{E}') \\~~\\
        \tilde{\mathbf{E}} \in \mathbb{R}^{T \times d}
        ''',
        "Transformer Encoder": r'''
        \mathbf{H}_{\text{Transformer}} = \text{TransformerEncoder}(\tilde{\mathbf{E}}, \text{mask}) \\~~\\
        \mathbf{H}_{\text{Transformer}} \in \mathbb{R}^{T \times d}
        ''',
        "Dropout (Post-Transformer)": r'''
        \tilde{\mathbf{H}}_{\text{Transformer}} = \text{Dropout}(\mathbf{H}_{\text{Transformer}}) \\~~\\
        \tilde{\mathbf{H}}_{\text{Transformer}} \in \mathbb{R}^{T \times d}
        ''',
        "Logits": r'''
        \mathbf{o} = \mathbf{W}_d \cdot \tilde{\mathbf{H}}_{\text{Transformer}} + \mathbf{b}_d \\~~\\
        \mathbf{o} \in \mathbb{R}^{T \times \text{output dim}}
        '''
        }
    }
}

# ----------------------
# Loading Function
# ----------------------

@st.cache_resource
def load_model(model_name, vocab):
    class SPositionalEncoding(nn.Module):
        def __init__(self, embed_size, max_len=5000):
            super(SPositionalEncoding, self).__init__()
            pe = torch.zeros(max_len, embed_size)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-np.log(10000.0) / embed_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            if embed_size % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return x
    class STransformerE(nn.Module):
        def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_dim, padding_idx, embedding_matrix, dropout=0.1):
            super(STransformerE, self).__init__()
            # Embedding Layer
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
            self.embedding.weight = nn.Parameter(embedding_matrix)
            self.embedding.weight.requires_grad = True
            # Positional Encoding Layer
            self.pos_encoder = SPositionalEncoding(embed_size)
            # Transformer Encoder Layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_size))
            # Dropout Layer
            self.dropout = nn.Dropout(dropout)
            # Fully Connected Layer
            self.fc = nn.Linear(embed_size, output_dim)
        def forward(self, x):
            embedded = self.embedding(x)
            embedded = self.pos_encoder(embedded)
            embedded = self.dropout(embedded)
            src_key_padding_mask = (x == self.embedding.padding_idx)
            transformer_output = self.transformer_encoder(
                embedded, 
                src_key_padding_mask=src_key_padding_mask
            )
            transformer_output = self.dropout(transformer_output)
            logits = self.fc(transformer_output)
            return logits
    try:
        embedding_matrix = torch.load(os.path.join("models", "embedding-matrix.pth"), weights_only=True, map_location=torch.device('cpu'))
        net = STransformerE(
            vocab_size=len(vocab),
            embed_size=200,
            num_heads=8,
            hidden_dim=512,
            num_layers=2,
            output_dim=47,
            padding_idx=0,
            embedding_matrix=embedding_matrix,
            dropout=0.16426146772147993
        )
        net.load_state_dict(torch.load(os.path.join("models", "model-state.pt"), weights_only=True, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    return net

@st.cache_data
def load_vocab(model_name):
    try:
        with open(os.path.join("models", "word2idx.json"), 'r') as json_file:
            word2idx = json.load(json_file)
        with open(os.path.join("models", "idx2pos.json"), 'r') as json_file:
            idx2pos = json.load(json_file)
            idx2pos = {int(k): v for k, v in idx2pos.items()}
        return word2idx, idx2pos
    except FileNotFoundError:
        st.error(f"Vocabulary file not found for {model_name}.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the vocabulary for {model_name}: {e}")
        st.stop()

# ----------------------
# Prediction Function
# ----------------------
def predict_pos_tag(net, word2idx, idx2pos, sequence):
    net.eval()
    pos_descriptions = {
        "CC": "Coordinating Conjunction",
        "CD": "Cardinal Number",
        "DT": "Determiner",
        "EX": "Existential 'There'",
        "FW": "Foreign Word",
        "IN": "Preposition or Subordinating Conjunction",
        "JJ": "Adjective",
        "JJR": "Adjective (Comparative)",
        "JJS": "Adjective (Superlative)",
        "LS": "List Item Marker",
        "MD": "Modal",
        "NN": "Noun (Singular or Mass)",
        "NNS": "Noun (Plural)",
        "NNP": "Proper Noun (Singular)",
        "NNPS": "Proper Noun (Plural)",
        "PDT": "Predeterminer",
        "POS": "Possessive Ending",
        "PRP": "Personal Pronoun",
        "PRP$": "Possessive Pronoun",
        "RB": "Adverb",
        "RBR": "Adverb (Comparative)",
        "RBS": "Adverb (Superlative)",
        "RP": "Particle",
        "SYM": "Symbol",
        "TO": "to",
        "UH": "Interjection",
        "VB": "Verb (Base Form)",
        "VBD": "Verb (Past Tense)",
        "VBG": "Verb (Gerund or Present Participle)",
        "VBN": "Verb (Past Participle)",
        "VBP": "Verb (Non-3rd-Person Singular Present)",
        "VBZ": "Verb (3rd Person Singular Present)",
        "WDT": "Wh-Determiner",
        "WP": "Wh-Pronoun",
        "WP$": "Possessive Wh-Pronoun",
        "WRB": "Wh-Adverb"
    }
    if isinstance(sequence, str):
        words = sequence.split()
    elif isinstance(sequence, list):
        words = sequence
    else:
        raise ValueError("Input sequence must be a string or list of words")
    words_lower = [word.lower() for word in words]
    word_indices = [word2idx.get(word, word2idx['<UNK>']) for word in words_lower]
    input_tensor = torch.tensor([word_indices], dtype=torch.long)
    lengths = torch.tensor([len(word_indices)], dtype=torch.long)
    with torch.no_grad():
        logits = net(input_tensor)
        predictions = torch.argmax(logits, dim=-1)
    predicted_pos_indices = predictions[0][:lengths[0]].cpu().numpy()
    predicted_pos_tags = [idx2pos[idx] for idx in predicted_pos_indices]
    word_pos_pairs = list(zip(words, predicted_pos_tags))
    tree = Tree('S', [Tree(pos, [word]) for word, pos in word_pos_pairs])
    ordered_unique_pos = []
    for pos in predicted_pos_tags:
        if pos not in ordered_unique_pos:
            ordered_unique_pos.append(pos)    
    description = {pos: pos_descriptions.get(pos, "Unknown POS tag") for pos in ordered_unique_pos}
    return tree, description

# ----------------------
# Page UI
# ----------------------
def main():
    st.title("POS Tagging")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    st.divider()
    
    word2idx, idx2pos = load_vocab(model)
    net = load_model(model, word2idx)
    
    st.subheader(model_info[model]["subheader"])
    
    # user_input = st.text_input("Enter Text Here:")
    # if st.button("Analyze"):
    #     if user_input.strip():
    #         with st.spinner('Analyzing...'):
    #             tree, description = predict_pos_tag(net, word2idx, idx2pos, user_input)
    #         st.code(TreePrettyPrinter(tree).text(), language="None")
    #         pos_df = pd.DataFrame(list(description.items()), columns=['POS Tag', 'Description'])
    #         st.table(pos_df.style.hide(axis="index"))
    #     else:
    #         st.warning("Please enter some text.")
    
    with st.form(key="pos_form"):
        user_input = st.text_input("Enter Text Here:")
        submit_button = st.form_submit_button(label="Tag")
        
        if submit_button:
            if user_input.strip():
                with st.spinner('Tagging...'):
                    tree, description = predict_pos_tag(net, word2idx, idx2pos, user_input)
                st.code(TreePrettyPrinter(tree).text(), language="None")
                pos_df = pd.DataFrame(list(description.items()), columns=['POS Tag', 'Description'])
                st.dataframe(pos_df, hide_index=True)
            else:
                st.warning("Please enter some text.")
    
    # st.divider()              
    st.feedback("thumbs")
    st.warning("""Check here for more details: [GitHub Repo🐙](https://github.com/verneylmavt/st-pos-tagging)""")
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass

if __name__ == "__main__":
    main()