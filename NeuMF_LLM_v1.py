# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :NeuMF_LLM.py
# @Time        :2025/1/21 20:46
# @Author      :JasonMa
# @ProjectName :neural_collaborative_filtering
import heapq
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# Dataset class for user-item interactions
class InteractionDataset(Dataset):
    def __init__(self, user_input, item_input, labels):
        self.user_input = torch.tensor(user_input, dtype=torch.long)
        self.item_input = torch.tensor(item_input, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_input[idx], self.item_input[idx], self.labels[idx]


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim, layers, reg_layers, reg_mf, user_texts=None, item_texts=None):
        super(NeuMF, self).__init__()

        if user_texts and item_texts:
            # Use pre-trained text model to generate embeddings
            self.text_model = AutoModel.from_pretrained("E:/Models/Qwen/Qwen2___5-3B-Instruct")
            self.tokenizer = AutoTokenizer.from_pretrained("E:/Models/Qwen/Qwen2___5-3B-Instruct")

            with torch.no_grad():
                user_embeddings = self._get_text_embeddings(user_texts)  # Shape: (num_users, embedding_dim)
                item_embeddings = self._get_text_embeddings(item_texts)  # Shape: (num_items, embedding_dim)

            # GMF Embedding Layers initialized with text embeddings
            self.mf_user_embedding = nn.Embedding.from_pretrained(user_embeddings, freeze=False)  # Shape: (num_users, embedding_dim)
            self.mf_item_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=False)  # Shape: (num_items, embedding_dim)

            # MLP Embedding Layers initialized with text embeddings
            self.mlp_user_embedding = nn.Embedding.from_pretrained(user_embeddings, freeze=False)  # Shape: (num_users, embedding_dim)
            self.mlp_item_embedding = nn.Embedding.from_pretrained(item_embeddings, freeze=False)  # Shape: (num_items, embedding_dim)
        else:
            # GMF Embedding Layers with default sizes
            self.mf_user_embedding = nn.Embedding(num_users, mf_dim)  # Shape: (num_users, mf_dim)
            self.mf_item_embedding = nn.Embedding(num_items, mf_dim)  # Shape: (num_items, mf_dim)

            # MLP Embedding Layers
            self.mlp_user_embedding = nn.Embedding(num_users, layers[0] // 2)  # Shape: (num_users, layers[0] // 2)
            self.mlp_item_embedding = nn.Embedding(num_items, layers[0] // 2)  # Shape: (num_items, layers[0] // 2)

        # MLP Layers (input size should be correctly adjusted)
        mlp_modules = []
        input_size = layers[0]  # Input size is the concatenated user and item embeddings
        for i in range(1, len(layers)):
            mlp_modules.append(nn.Linear(input_size, layers[i]))  # Each layer's input and output size should match
            mlp_modules.append(nn.ReLU())  # Apply ReLU for activation
            input_size = layers[i]  # Update input size for the next layer
        self.mlp = nn.Sequential(*mlp_modules)

        # Final Prediction Layer (input size should match mf_dim + layers[-1])
        self.predict_layer = nn.Linear(mf_dim + layers[-1], 1)  # Shape: (batch_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        # GMF Part: Element-wise multiplication of user and item embeddings
        mf_user_latent = self.mf_user_embedding(user_input)  # Shape: (batch_size, mf_dim)
        mf_item_latent = self.mf_item_embedding(item_input)  # Shape: (batch_size, mf_dim)
        mf_vector = mf_user_latent * mf_item_latent  # Element-wise multiplication (Shape: (batch_size, mf_dim))

        # MLP Part: Concatenate user and item embeddings and pass through MLP layers
        mlp_user_latent = self.mlp_user_embedding(user_input)  # Shape: (batch_size, layers[0] // 2)
        mlp_item_latent = self.mlp_item_embedding(item_input)  # Shape: (batch_size, layers[0] // 2)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)  # Concatenate (Shape: (batch_size, layers[0]))
        mlp_vector = self.mlp(mlp_vector)  # Pass through MLP layers (Shape: (batch_size, layers[-1]))

        # Concatenate MF and MLP parts to form the final prediction vector
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)  # Shape: (batch_size, mf_dim + layers[-1])

        # Final Prediction
        prediction = self.sigmoid(self.predict_layer(predict_vector))  # Shape: (batch_size, 1)
        return prediction

    def _get_text_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Using mean pooling for text embeddings


# Load dataset from files
def load_dataset(train_file, test_file, negative_file):
    train_data = {}
    test_ratings = []
    test_negatives = []
    all_users = set()
    all_items = set()

    # Load train.rating
    with open(train_file, 'r') as f:
        for line in f:
            user, item, _ = line.strip().split('\t')[:3]
            user, item = int(user), int(item)
            train_data[(user, item)] = 1
            all_users.add(user)
            all_items.add(item)

    # Load test.rating
    with open(test_file, 'r') as f:
        for line in f:
            user, item, _ = line.strip().split('\t')[:3]
            user, item = int(user), int(item)
            test_ratings.append((user, item))
            all_users.add(user)
            all_items.add(item)

    # Load test.negative
    with open(negative_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            user, pos_item = eval(parts[0])  # (userId, positiveItemId)
            negatives = list(map(int, parts[1:]))
            test_negatives.append((user, pos_item, negatives))
            all_users.add(user)
            all_items.update(negatives)

    num_users = max(all_users) + 1
    num_items = max(all_items) + 1

    return train_data, test_ratings, test_negatives, num_users, num_items


# Generate training instances
def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # Positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # Negative instances
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


# Evaluate the model performance
def evaluate_model(model, test_ratings, test_negatives, top_k, device):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation.

    Args:
        model: Trained PyTorch model.
        test_ratings: List of tuples [(user, item), ...] for test ratings.
        test_negatives: List of tuples [(user, pos_item, [neg_item1, neg_item2, ...])].
        top_k: Number of items to recommend.
        device: Device to run evaluation (CPU or GPU).

    Returns:
        hits: List of Hit Ratio scores for each test case.
        ndcgs: List of NDCG scores for each test case.
    """
    model.eval()
    hits, ndcgs = [], []

    for idx in range(len(test_ratings)):
        user, gt_item = test_ratings[idx]
        neg_items = test_negatives[idx][2]
        items = neg_items + [gt_item]

        # Create input tensors
        user_tensor = torch.tensor([user] * len(items), dtype=torch.long, device=device)
        item_tensor = torch.tensor(items, dtype=torch.long, device=device)

        # Get prediction scores
        with torch.no_grad():
            predictions = model(user_tensor, item_tensor).squeeze().cpu().numpy()

        # Map item scores and rank them
        map_item_score = {item: score for item, score in zip(items, predictions)}
        ranklist = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)

        # Evaluate metrics
        hits.append(get_hit_ratio(ranklist, gt_item))
        ndcgs.append(get_ndcg(ranklist, gt_item))

    return hits, ndcgs


def get_hit_ratio(ranklist, gt_item):
    """Compute Hit Ratio."""
    if gt_item in ranklist:
        return 1
    return 0


def get_ndcg(ranklist, gt_item):
    """Compute NDCG."""
    if gt_item in ranklist:
        index = ranklist.index(gt_item)
        return math.log(2) / math.log(index + 2)
    return 0


# Training loop
def train_model(model, train_loader, optimizer, criterion, epochs, device, test_ratings, test_negatives, top_k):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, label in train_loader:
            user, item, label = user.to(device), item.to(device), label.to(device)

            # Forward pass
            prediction = model(user, item).squeeze()
            loss = criterion(prediction, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate model
        hits, ndcgs = evaluate_model(model, test_ratings, test_negatives, top_k, device)
        avg_hit = np.mean(hits)
        avg_ndcg = np.mean(ndcgs)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, HR: {avg_hit:.4f}, NDCG: {avg_ndcg:.4f}")


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = 0
    num_negatives = 4
    learning_rate = 0.001
    batch_size = 256
    epochs = 20
    top_k = 10

    # File paths
    train_file = "Data/ml-1m.train.rating"
    test_file = "Data/ml-1m.test.rating"
    negative_file = "Data/ml-1m.test.negative"

    # Load dataset
    train_data, test_ratings, test_negatives, num_users, num_items = load_dataset(train_file, test_file, negative_file)
    user_input, item_input, labels = get_train_instances(train_data, num_negatives, num_items)
    train_dataset = InteractionDataset(user_input, item_input, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Example user and item text data (replace with actual text if available)
    user_texts = [f"User {i}" for i in range(num_users)]
    item_texts = [f"Item {i}" for i in range(num_items)]

    # Model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, user_texts, item_texts)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Train the model
    train_model(model, train_loader, optimizer, criterion, epochs, device, test_ratings, test_negatives, top_k)
