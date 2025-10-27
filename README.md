🗣️ Urdu Conversational Chatbot using Transformer
🚀 Project 02 — NLP (Group Assignment)

This project implements a custom Urdu conversational chatbot built entirely from scratch using a Transformer Encoder-Decoder architecture. The chatbot understands and generates Urdu text, leveraging multi-head attention for contextual understanding.
It also includes a Streamlit interface for real-time interaction.

🔗 Medium Blog: Building an Urdu Conversational Chatbot using Transformers

📂 GitHub Repository: Assignment-2_Urdu-Conversational-Chatbot

📘 Overview

The goal of this project is to design and train a Transformer-based chatbot capable of generating coherent and context-aware Urdu responses.
The model was trained on a dataset of Urdu conversations and deployed with a simple Streamlit interface for demonstration.

🧠 Model Architecture

Transformer Encoder-Decoder implemented from scratch (PyTorch)

Multi-Head Attention for contextual understanding

Positional Encoding for sequence order awareness

Feed-Forward Networks for feature transformation

Greedy & Beam Search decoding for response generation

⚙️ Features

✅ Real-time Urdu chat via Streamlit
✅ Right-to-left text rendering support
✅ Custom-trained vocabulary & SentencePiece tokenizer
✅ Manual model loading (.pth and .model files)
✅ Supports both Greedy and Beam Search decoding
✅ Evaluation using BLEU, ROUGE-L, and chrF metrics

📂 Dataset

Used dataset: Urdu Conversational Dataset (20,000 samples)

🧾 Preprocessing Steps

Normalization — Removing diacritics, standardizing Alef & Yeh forms

Tokenization — Using SentencePiece tokenizer

Vocabulary Creation — Based on frequency threshold

Dataset Split:

Train: 80%

Validation: 10%

Test: 10%

🔧 Hyperparameters
Parameter	Value
Embedding Dim	256
Heads	2
Encoder Layers	2
Decoder Layers	2
Dropout	0.1
Batch Size	32
Learning Rate	1e-4
Optimizer	Adam
🧩 Evaluation Metrics
Metric	Description
BLEU	Measures overlap between generated and reference text
ROUGE-L	Evaluates longest common subsequence similarity
chrF	Character n-gram F-score (captures word variations)
Perplexity	Measures model confidence in its predictions
🧑‍💻 Authors

Muhammad Abdul Muizz
Medium | GitHub
