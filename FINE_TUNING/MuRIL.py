"""
MuRIL Fine-tuning for Manipuri (Low-Resource) NLU Tasks
Tasks: Text Classification, Headline Classification, NLI, Text Similarity, Retrieval
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple
import random
import json
import pandas as pd

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# LOADING DATA FROM FILES:

def read_data():
    # Text Classification
    # Load the JSON file
    with open(r"E:\NLU\content_classification\train_content_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ✅ Works when the top-level structure is a list
    text_classification_data = [(item['text'], item['label']) for item in data]

    # Get all unique labels
    unique_labels = sorted(set(label for _, label in text_classification_data))

    # Create label → numeric ID mapping
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    # Replace labels with numeric IDs
    text_classification_data = [(text, label2id[label]) for text, label in text_classification_data]

    # Optional: show the mapping and a few examples
    print("✅ Label-to-ID mapping Text Classification:")
    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    # Headline Classification
    df_text = pd.read_csv(r"E:\NLU\News_headlineclassification_train.csv")
    # headline_classification_data = list(zip(df_headline['headline'], df_headline['category']))
    # Create a mapping from label → numeric ID
    label2id = {label: idx for idx, label in enumerate(sorted(df_text['Category'].unique()))}

    # Replace text labels with numeric IDs
    df_text['label_id'] = df_text['Category'].map(label2id)

    # Create list of tuples (text, numeric_label)
    headline_classification_data = list(zip(df_text['Headlines'], df_text['label_id']))

    # Optional: print mapping and sample data
    print("✅ Label-to-ID mapping headline_classification_data:")
    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    # NLI
    # df_nli = pd.read_csv('nli.csv')
    # nli_data = list(zip(df_nli['Premise'], df_nli['Hypothesis'], df_nli['Label']))
    # Load CSV file
    df_nli = pd.read_csv(r"E:\NLU\NLI\train_nli1.csv")

    # Create label → numeric ID mapping
    label2id = {label: idx for idx, label in enumerate(sorted(df_nli['Label'].unique()))}

    # Add numeric label column
    df_nli['Label_id'] = df_nli['Label'].map(label2id)

    # Create list of tuples (Premise, Hypothesis, numeric_label)
    nli_data = list(zip(df_nli['Premise'], df_nli['Hypothesis'], df_nli['Label_id']))

    # Optional: print mapping and sample data
    print("✅ Label-to-ID mapping:")
    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    # Text Similarity
    df_sim = pd.read_csv(r"E:\NLU\sentence_similarity\semanticsimilarity_train.csv")
    similarity_data = list(zip(df_sim['SENTENCE1'], df_sim['SENTENCE2'], df_sim['SCORE']))

    # Retrieval
    # df_ret = pd.read_csv('retrieval.csv')
    # # Assuming negatives are in columns: neg1, neg2, neg3
    # retrieval_data = [
    #     (row['query'], row['positive'], [row['neg1'], row['neg2'], row['neg3']])
    #     for _, row in df_ret.iterrows()
    # ]
    # Load the retrieval JSON file
    with open(r"E:\NLU\retrieval_train.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build list of tuples: (query, positive, [random, tfidf, bm25])
    retrieval_data = [
        (
            item['query'],
            item['positive'],
            [item['negatives']['random'], item['negatives']['tfidf'], item['negatives']['bm25']]
        )
        for item in data
    ]
    return {
        'text_classification': text_classification_data,
        'headline_classification': headline_classification_data,
        'nli': nli_data,
        'similarity': similarity_data,
        'retrieval': retrieval_data
    }


# ===========================
# DATASET CLASSES
# ===========================

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = str(self.premises[idx])
        hypothesis = str(self.hypotheses[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SimilarityDataset(Dataset):
    def __init__(self, text1_list, text2_list, scores, tokenizer, max_length=128):
        self.text1_list = text1_list
        self.text2_list = text2_list
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text1_list)

    def __getitem__(self, idx):
        text1 = str(self.text1_list[idx])
        text2 = str(self.text2_list[idx])
        score = self.scores[idx]

        encoding = self.tokenizer(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }


class RetrievalDataset(Dataset):
    def __init__(self, queries, positives, negatives, tokenizer, max_length=128):
        self.queries = queries
        self.positives = positives
        self.negatives = negatives  # List of lists, each containing 3 negatives
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        positive = str(self.positives[idx])
        negative_samples = [str(neg) for neg in self.negatives[idx]]

        # Encode query
        query_enc = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode positive
        pos_enc = self.tokenizer(
            positive,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode negatives
        neg_encs = []
        for neg in negative_samples:
            neg_enc = self.tokenizer(
                neg,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            neg_encs.append({
                'input_ids': neg_enc['input_ids'].flatten(),
                'attention_mask': neg_enc['attention_mask'].flatten()
            })

        return {
            'query_input_ids': query_enc['input_ids'].flatten(),
            'query_attention_mask': query_enc['attention_mask'].flatten(),
            'pos_input_ids': pos_enc['input_ids'].flatten(),
            'pos_attention_mask': pos_enc['attention_mask'].flatten(),
            'neg_input_ids': torch.stack([neg['input_ids'] for neg in neg_encs]),
            'neg_attention_mask': torch.stack([neg['attention_mask'] for neg in neg_encs])
        }


# ===========================
# MODEL ARCHITECTURES
# ===========================

class MuRILClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class MuRILSimilarity(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        score = self.regressor(pooled_output)
        return score


class MuRILRetrieval(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def encode(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)


# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_classification(model, train_loader, val_loader, device, epochs=10, lr=2e-5):
    """Train classification model"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1*total_steps,
        num_training_steps=total_steps
    )

    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")

    return model


def train_similarity(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    """Train similarity model (regression)"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1*total_steps,
        num_training_steps=total_steps
    )

    best_val_mse = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            scores = model(input_ids, attention_mask).squeeze()
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                scores = model(input_ids, attention_mask).squeeze()
                loss = criterion(scores, labels)

                val_loss += loss.item()
                val_preds.extend(scores.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_mse = mean_squared_error(val_labels, val_preds)

        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, MSE: {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"New best validation MSE: {best_val_mse:.4f}")

    return model


def train_retrieval(model, train_loader, val_loader, device, epochs=3, lr=2e-5, margin=0.5):
    """Train retrieval model with triplet loss"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1*total_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            query_ids = batch['query_input_ids'].to(device)
            query_mask = batch['query_attention_mask'].to(device)
            pos_ids = batch['pos_input_ids'].to(device)
            pos_mask = batch['pos_attention_mask'].to(device)
            neg_ids = batch['neg_input_ids'].to(device)  # [batch, num_neg, seq_len]
            neg_mask = batch['neg_attention_mask'].to(device)

            optimizer.zero_grad()

            # Encode query and positive
            query_emb = model.encode(query_ids, query_mask)
            pos_emb = model.encode(pos_ids, pos_mask)

            # Encode negatives (loop through each negative)
            batch_size = query_ids.size(0)
            num_negatives = neg_ids.size(1)

            total_loss = 0
            for i in range(num_negatives):
                neg_emb = model.encode(neg_ids[:, i, :], neg_mask[:, i, :])
                loss = criterion(query_emb, pos_emb, neg_emb)
                total_loss += loss

            total_loss = total_loss / num_negatives
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                query_ids = batch['query_input_ids'].to(device)
                query_mask = batch['query_attention_mask'].to(device)
                pos_ids = batch['pos_input_ids'].to(device)
                pos_mask = batch['pos_attention_mask'].to(device)
                neg_ids = batch['neg_input_ids'].to(device)
                neg_mask = batch['neg_attention_mask'].to(device)

                query_emb = model.encode(query_ids, query_mask)
                pos_emb = model.encode(pos_ids, pos_mask)

                batch_size = query_ids.size(0)
                num_negatives = neg_ids.size(1)

                total_loss = 0
                for i in range(num_negatives):
                    neg_emb = model.encode(neg_ids[:, i, :], neg_mask[:, i, :])
                    loss = criterion(query_emb, pos_emb, neg_emb)
                    total_loss += loss

                total_loss = total_loss / num_negatives
                val_loss += total_loss.item()

        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")

    return model


# ===========================
# MAIN EXECUTION
# ===========================

def main():
    # Configuration
    MODEL_NAME = "google/muril-base-cased"
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("\n" + "="*50)
    print("Loading MuRIL tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create dummy datasets
    print("\n" + "="*50)
    print("Reading data....")
    datasets = read_data()

    # ===========================
    # Task 1: Text Classification (3 classes)
    # ===========================
    print("\n" + "="*50)
    print("TASK 1: Text Classification (3 classes)")
    print("="*50)

    texts = [item[0] for item in datasets['text_classification']]
    labels = [item[1] for item in datasets['text_classification']]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    text_clf_model = MuRILClassifier(MODEL_NAME, num_classes=3)
    text_clf_model = train_classification(text_clf_model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Task 2: Headline Classification (4 classes)
    # ===========================
    print("\n" + "="*50)
    print("TASK 2: Headline Classification (4 classes)")
    print("="*50)

    headlines = [item[0] for item in datasets['headline_classification']]
    h_labels = [item[1] for item in datasets['headline_classification']]

    train_hl, val_hl, train_hl_labels, val_hl_labels = train_test_split(
        headlines, h_labels, test_size=0.2, random_state=42
    )

    train_hl_dataset = ClassificationDataset(train_hl, train_hl_labels, tokenizer, MAX_LENGTH)
    val_hl_dataset = ClassificationDataset(val_hl, val_hl_labels, tokenizer, MAX_LENGTH)

    train_hl_loader = DataLoader(train_hl_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_hl_loader = DataLoader(val_hl_dataset, batch_size=BATCH_SIZE)

    headline_clf_model = MuRILClassifier(MODEL_NAME, num_classes=4)
    headline_clf_model = train_classification(headline_clf_model, train_hl_loader, val_hl_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Task 3: NLI (3 classes)
    # ===========================
    print("\n" + "="*50)
    print("TASK 3: Natural Language Inference (NLI)")
    print("="*50)

    premises = [item[0] for item in datasets['nli']]
    hypotheses = [item[1] for item in datasets['nli']]
    nli_labels = [item[2] for item in datasets['nli']]

    train_prem, val_prem, train_hyp, val_hyp, train_nli_labels, val_nli_labels = train_test_split(
        premises, hypotheses, nli_labels, test_size=0.2, random_state=42
    )

    # Split train data separately from validation
    train_prem_split = [train_prem[i] for i in range(len(train_prem))]
    train_hyp_split = [train_hyp[i] for i in range(len(train_hyp))]

    train_nli_dataset = NLIDataset(train_prem_split, train_hyp_split, train_nli_labels, tokenizer, MAX_LENGTH)
    val_nli_dataset = NLIDataset(val_prem, val_hyp, val_nli_labels, tokenizer, MAX_LENGTH)

    train_nli_loader = DataLoader(train_nli_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_nli_loader = DataLoader(val_nli_dataset, batch_size=BATCH_SIZE)

    nli_model = MuRILClassifier(MODEL_NAME, num_classes=3)
    nli_model = train_classification(nli_model, train_nli_loader, val_nli_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Task 4: Text Similarity (Regression)
    # ===========================
    print("\n" + "="*50)
    print("TASK 4: Text Similarity (Regression)")
    print("="*50)

    text1_list = [item[0] for item in datasets['similarity']]
    text2_list = [item[1] for item in datasets['similarity']]
    sim_scores = [item[2] for item in datasets['similarity']]

    train_t1, val_t1, train_t2, val_t2, train_scores, val_scores = train_test_split(
        text1_list, text2_list, sim_scores, test_size=0.2, random_state=42
    )

    train_sim_dataset = SimilarityDataset(train_t1, train_t2, train_scores, tokenizer, MAX_LENGTH)
    val_sim_dataset = SimilarityDataset(val_t1, val_t2, val_scores, tokenizer, MAX_LENGTH)

    train_sim_loader = DataLoader(train_sim_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_sim_loader = DataLoader(val_sim_dataset, batch_size=BATCH_SIZE)

    similarity_model = MuRILSimilarity(MODEL_NAME)
    similarity_model = train_similarity(similarity_model, train_sim_loader, val_sim_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Task 5: Retrieval
    # ===========================
    print("\n" + "="*50)
    print("TASK 5: Retrieval (Query, Positive, Negatives)")
    print("="*50)

    queries = [item[0] for item in datasets['retrieval']]
    positives = [item[1] for item in datasets['retrieval']]
    negatives = [item[2] for item in datasets['retrieval']]

    train_q, val_q, train_pos, val_pos, train_neg, val_neg = train_test_split(
        queries, positives, negatives, test_size=0.2, random_state=42
    )

    train_ret_dataset = RetrievalDataset(train_q, train_pos, train_neg, tokenizer, MAX_LENGTH)
    val_ret_dataset = RetrievalDataset(val_q, val_pos, val_neg, tokenizer, MAX_LENGTH)

    train_ret_loader = DataLoader(train_ret_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_ret_loader = DataLoader(val_ret_dataset, batch_size=BATCH_SIZE)

    retrieval_model = MuRILRetrieval(MODEL_NAME)
    retrieval_model = train_retrieval(retrieval_model, train_ret_loader, val_ret_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Save Models
    # ===========================
    print("\n" + "="*50)
    print("Saving models...")
    torch.save(text_clf_model.state_dict(), '.venv/MuRIL_full/text_classification_model.pt')
    torch.save(headline_clf_model.state_dict(), '.venv/MuRIL_full/headline_classification_model.pt')
    torch.save(nli_model.state_dict(), '.venv/MuRIL_full/nli_model.pt')
    torch.save(similarity_model.state_dict(), '.venv/MuRIL_full/similarity_model.pt')
    torch.save(retrieval_model.state_dict(), '.venv/MuRIL_full/retrieval_model.pt')
    print("All models saved successfully!")

    print("\n" + "="*50)
    print("Training completed for all tasks!")
    print("="*50)


if __name__ == "__main__":
    main()

