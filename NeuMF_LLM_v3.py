#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# @Project ：neural_collaborative_filtering
# @File    ：NeuMF_LLM_v3.py
# @Author  ：majinjin
# @Date    ：2025/1/23
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import heapq
from transformers import AutoModel, AutoTokenizer

class InteractionDataset(Dataset):
    def __init__(self, user_texts, item_texts, labels, tokenizer, max_len=128):
        self.user_tokens = tokenizer(user_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.item_tokens = tokenizer(item_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_input = {key: val[idx] for key, val in self.user_tokens.items()}
        item_input = {key: val[idx] for key, val in self.item_tokens.items()}
        label = self.labels[idx]
        return user_input, item_input, label

class NeuMF(nn.Module):
    def __init__(self, text_model_name, mf_dim, layers):
        super(NeuMF, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)

        text_embedding_dim = self.text_model.config.hidden_size

        self.mf_user_projection = nn.Linear(text_embedding_dim, mf_dim)
        self.mf_item_projection = nn.Linear(text_embedding_dim, mf_dim)

        self.mlp_user_projection = nn.Linear(text_embedding_dim, layers[0] // 2)
        self.mlp_item_projection = nn.Linear(text_embedding_dim, layers[0] // 2)

        mlp_modules = []
        for i in range(1, len(layers)):
            mlp_modules.append(nn.Linear(layers[i - 1], layers[i]))
            mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)

        self.predict_layer = nn.Linear(mf_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        user_embedding = self.text_model(**user_input).last_hidden_state.mean(dim=1)
        item_embedding = self.text_model(**item_input).last_hidden_state.mean(dim=1)

        mf_user_latent = self.mf_user_projection(user_embedding)
        mf_item_latent = self.mf_item_projection(item_embedding)
        mf_vector = mf_user_latent * mf_item_latent

        mlp_user_latent = self.mlp_user_projection(user_embedding)
        mlp_item_latent = self.mlp_item_projection(item_embedding)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.sigmoid(self.predict_layer(predict_vector))
        return prediction

def load_text_dataset(train_file, test_file, negative_file):
    train_users, train_items, train_labels = [], [], []
    test_users, test_items = [], []
    test_negatives = []

    with open(train_file, 'r') as f:
        for line in f:
            user_name, item_name, label = line.strip().split('\t')[:3]
            train_users.append(user_name)
            train_items.append(item_name)
            train_labels.append(max(0.0, min(1.0, float(label))))

    with open(test_file, 'r') as f:
        for line in f:
            user_name, item_name, _ = line.strip().split('\t')[:3]
            test_users.append(user_name)
            test_items.append(item_name)

    with open(negative_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user_name, pos_item_name = eval(parts[0])
            negatives = parts[1:]
            test_negatives.append((user_name, pos_item_name, negatives))

    return (train_users, train_items, train_labels), (test_users, test_items, test_negatives)

def evaluate_model(model, test_users, test_items, test_negatives, tokenizer, top_k, device):
    model.eval()
    hits, ndcgs = [], []

    for idx in range(len(test_users)):
        user_name = test_users[idx]
        gt_item_name = test_items[idx]

        # Tokenize user and ground-truth item
        user_input = tokenizer(user_name, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
        gt_item_input = tokenizer(gt_item_name, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)

        # Tokenize negative samples
        negatives = test_negatives[idx][2]
        negative_inputs = tokenizer(negatives, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)

        # Debug: Print shapes
        print(f"gt_item_input shape: {gt_item_input['input_ids'].shape}")
        print(f"negative_inputs shape: {negative_inputs['input_ids'].shape}")

        # Adjust tensor dimensions for concatenation
        for key in gt_item_input:
            gt_item_input[key] = gt_item_input[key].expand(len(negatives) + 1, -1)

        # Debug: Print shapes after expansion
        print(f"gt_item_input shape after expansion: {gt_item_input['input_ids'].shape}")

        all_item_inputs = {key: torch.cat([gt_item_input[key], negative_inputs[key]], dim=0) for key in gt_item_input}

        with torch.no_grad():
            user_embedding = model.text_model(**user_input).last_hidden_state.mean(dim=1)
            item_embeddings = model.text_model(**all_item_inputs).last_hidden_state.mean(dim=1)

            mf_user_latent = model.mf_user_projection(user_embedding)
            mf_item_latents = model.mf_item_projection(item_embeddings)
            mf_vector = mf_user_latent * mf_item_latents

            mlp_user_latent = model.mlp_user_projection(user_embedding)
            mlp_item_latents = model.mlp_item_projection(item_embeddings)
            mlp_vector = torch.cat([mlp_user_latent.repeat(len(item_embeddings), 1), mlp_item_latents], dim=-1)
            mlp_vector = model.mlp(mlp_vector)

            predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
            predictions = model.sigmoid(model.predict_layer(predict_vector)).squeeze()

        # Get top-k recommendations
        _, indices = torch.topk(predictions, top_k)
        recommended_items = indices.cpu().numpy()

        # Calculate HR and NDCG
        hr = int(0 in recommended_items)
        ndcg = math.log(2) / math.log(np.where(recommended_items == 0)[0][0] + 2) if hr else 0

        hits.append(hr)
        ndcgs.append(ndcg)

    return np.mean(hits), np.mean(ndcgs)


if __name__ == "__main__":
    text_model_name = "/root/autodl-fs/models/meta-llama/Meta-Llama-3-8B-Instruct"
    mf_dim = 8
    layers = [64, 32, 16, 8]
    learning_rate = 0.001
    batch_size = 16
    epochs = 3
    top_k = 10

    train_file = "Data/ml-s.train.rating"
    test_file = "Data/ml-s.test.rating"
    negative_file = "Data/ml-s.test.negative"

    (train_users, train_items, train_labels), (test_users, test_items, test_negatives) = load_text_dataset(train_file, test_file, negative_file)

    tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = InteractionDataset(train_users, train_items, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = NeuMF(text_model_name, mf_dim, layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_input, item_input, label in train_loader:
            user_input = {key: val.to(device) for key, val in user_input.items()}
            item_input = {key: val.to(device) for key, val in item_input.items()}
            label = label.to(device)

            prediction = model(user_input, item_input).squeeze()
            loss = criterion(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        hr, ndcg = evaluate_model(model, test_users, test_items, test_negatives, tokenizer, top_k, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, HR: {hr:.4f}, NDCG: {ndcg:.4f}")