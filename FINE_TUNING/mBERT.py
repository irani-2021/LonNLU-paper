"""
IndicBERT Fine-tuning for Manipuri (Low-Resource) NLU Tasks
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
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# ===========================
# DUMMY DATA GENERATION
# ===========================

def create_dummy_datasets():
    """Create dummy datasets for all tasks in Manipuri-like text"""

    # Text Classification (3 classes: news, opinion, entertainment)
    text_classification_data = [
        ("মণিপুর সরকারনা অমুক থৌরম তৌবা হৌদোকখ্রে", 0),  # news
        ("নুপীশিং মমিং কয়া অমদি ইন্দুনা য়াওরে", 1),  # opinion
        ("লাইনা থমজবা অসি মরম ওইদুনা য়াম্মদগী য়াম্না ফবা ওইখি", 2),  # entertainment
        ("শরুক নোংমা অমদি হকচাং লৈবাক তরেত য়েংশিনবা তৌগদবনি", 0),
        ("অমুক শেম নৈদবা নুপী অমত্তা থৌরাং মখুদা থাদোকপা হৌরক্লে", 1),
        ("শাই থোকহনবা অসি নুংঙাইবা অমসুং পুক্নিং থৌগৎপা অমনি", 2),
    ] * 50  # Repeat for more samples

    # Headline Classification (4 classes: politics, sports, culture, business)
    headline_classification_data = [
        ("মন্ত্রীনা অরাং সিং খুনবা য়াবা অমত্তগী পোরশু লৌবা", 0),  # politics
        ("ফুটবল টিমনা শাংলৈ তরেত লান্নরিবা মতাংদা মরুওইনা হান্না", 1),  # sports
        ("লাই হরাওবা পান্দমগী খুদোংথীবা ফংবা", 2),  # culture
        ("মার্কেট অমত্তদা কম্পেনী অমনা শরুক অমদি ব্যবসায়সু হৌদোক্লে", 3),  # business
        ("লাল সাহেবনা মীয়াম্বু মাংফম তরেত লৌথোক্লে", 0),
        ("ক্রিকেট টিমনা চহি অশুম্বদা থুংলবা মতুংদা ওইনা", 1),
        ("লাইনা থোংবা অমদি মণিপুরগী সংস্কৃতি লমদম", 2),
        ("অর্থনীতি থারাং মমাং অমত্তদা অমুক ফংবসু হায়রবা", 3),
    ] * 40

    # NLI (3 classes: entailment, contradiction, neutral)
    nli_data = [
        ("নুপী অমনা চিংজাও চানরি", "মী অমনা চিংজাও লাইরি", 0),  # entailment
        ("থা তারি", "উম তারি", 1),  # contradiction
        ("হিংচাবা অমা পানবদা ফজরি", "শিং পান্থোকপা য়াম্মনা ফবনি", 2),  # neutral
        ("ঙসি নুমিৎ অমনি", "অহিংবা নুংথিল য়াওরি", 1),
        ("মনুং অমা চক চানরি", "মনুং তা চানরি", 0),
        ("পুং সানা থাংখি", "থাবক অসি শাংনা লোইখি", 2),
    ] * 50

    # Text Similarity (regression, 0-5 scale)
    similarity_data = [
        ("মণিপুর অমা ফবা লৈবাকনি", "মণিপুর নুংঙাইতবা রাজ্য অমনি", 4.5),
        ("নোংমা খুনবা য়াওই", "নোংমা অমদি য়াংনা খুনবা তাই", 3.8),
        ("চা থক্লকপা পম্ব", "মোটর গাড়ী অমনা চৎথরকপা", 0.5),
        ("লাইনা অমদি থমোই থবক তৌবা", "লাইনা থবক অমদি থমোই চাবা", 4.2),
        ("হিংচাবা অমা লাক্লি", "নুপা অমনা লৈফম লাক্লি", 2.1),
        ("লাইনা অমদি থমজবা ফবনি", "খুদোল থমজবা য়াম্না নুংঙাইবনি", 3.5),
    ] * 50

    # Retrieval (query, positive, 3 negatives)
    retrieval_data = [
        (
            "মণিপুরগী থৌরম কনবু উথোরকপা",
            "মণিপুরদা ঐখোয়না লাই হরাওবা অমদি লাই হাওবা তৌয়",
            ["ফুটবল য়েংবা অসি ফবনি", "চিংজাও য়াম্না ফবনি", "ব্যবসায় তৌবা য়াম্না ফবা"]
        ),
        (
            "মণিপুরীদা চাক চাবা কঙবু উনবা",
            "মণিপুরী থোংদা চাক চাবা কঙবু 'চাক চানবা' হায়না হায়",
            ["পোতশেং চাক চাবা", "শরুক লৈবা মফম", "নুংঙাইতবা পুং অমা"]
        ),
        (
            "ঐনা লাইনা থমজবা পম্বগে",
            "লাইনা থমজবা অসি পুক্নিং থৌগৎপগী থবক অমনি",
            ["গাড়ী চংহনবা য়াই", "লম্বী থোংবা পল্লী", "মার্কেটতা চৎপা হৌগদবনি"]
        ),
    ] * 30

    return {
        'text_classification': text_classification_data,
        'headline_classification': headline_classification_data,
        'nli': nli_data,
        'similarity': similarity_data,
        'retrieval': retrieval_data
    }


# LOADING DATA FROM FILES:

def read_data():
    # Text Classification
    # Load the JSON file
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/content_classification/train_content_dataset.json", 'r', encoding='utf-8') as f:
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
    df_text = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/News_headlineclassification_train.csv")
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
    df_nli = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/NLI/train_nli1.csv")

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
    df_sim = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/sentence_similarity/semanticsimilarity_train.csv")
    similarity_data = list(zip(df_sim['SENTENCE1'], df_sim['SENTENCE2'], df_sim['SCORE']))

    # Retrieval
    # df_ret = pd.read_csv('retrieval.csv')
    # # Assuming negatives are in columns: neg1, neg2, neg3
    # retrieval_data = [
    #     (row['query'], row['positive'], [row['neg1'], row['neg2'], row['neg3']])
    #     for _, row in df_ret.iterrows()
    # ]
    # Load the retrieval JSON file
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/retrieval_train.json", 'r', encoding='utf-8') as f:
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

class IndicBERTClassifier(nn.Module):
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


class IndicBERTSimilarity(nn.Module):
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


class IndicBERTRetrieval(nn.Module):
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
    """
    Train retrieval model with triplet loss (supports multi-GPU DataParallel models)
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")

        # ------------------------------
        # TRAINING
        # ------------------------------
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            query_ids = batch['query_input_ids'].to(device)
            query_mask = batch['query_attention_mask'].to(device)
            pos_ids = batch['pos_input_ids'].to(device)
            pos_mask = batch['pos_attention_mask'].to(device)
            neg_ids = batch['neg_input_ids'].to(device)  # [batch, num_neg, seq_len]
            neg_mask = batch['neg_attention_mask'].to(device)

            optimizer.zero_grad()

            # ✅ Handle DataParallel: access underlying model if wrapped
            core_model = model.module if isinstance(model, torch.nn.DataParallel) else model

            # Encode anchor and positive
            query_emb = core_model.encode(query_ids, query_mask)
            pos_emb = core_model.encode(pos_ids, pos_mask)

            batch_size = query_ids.size(0)
            num_negatives = neg_ids.size(1)
            total_loss = 0.0

            # Loop through negatives and compute triplet loss
            for i in range(num_negatives):
                neg_emb = core_model.encode(neg_ids[:, i, :], neg_mask[:, i, :])
                loss = criterion(query_emb, pos_emb, neg_emb)
                total_loss += loss

            total_loss = total_loss / num_negatives
            total_loss.backward()

            # ✅ Optional: clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # ------------------------------
        # VALIDATION
        # ------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                query_ids = batch['query_input_ids'].to(device)
                query_mask = batch['query_attention_mask'].to(device)
                pos_ids = batch['pos_input_ids'].to(device)
                pos_mask = batch['pos_attention_mask'].to(device)
                neg_ids = batch['neg_input_ids'].to(device)
                neg_mask = batch['neg_attention_mask'].to(device)

                core_model = model.module if isinstance(model, torch.nn.DataParallel) else model

                query_emb = core_model.encode(query_ids, query_mask)
                pos_emb = core_model.encode(pos_ids, pos_mask)

                batch_size = query_ids.size(0)
                num_negatives = neg_ids.size(1)
                total_loss = 0.0

                for i in range(num_negatives):
                    neg_emb = core_model.encode(neg_ids[:, i, :], neg_mask[:, i, :])
                    loss = criterion(query_emb, pos_emb, neg_emb)
                    total_loss += loss

                total_loss = total_loss / num_negatives
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  🏋️ Train Loss: {avg_train_loss:.4f}")
        print(f"  🧪 Val Loss:   {avg_val_loss:.4f}")

        # ------------------------------
        # SAVE BEST MODEL
        # ------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✅ New best validation loss: {best_val_loss:.4f}")

    print(f"\n✅ Training complete. Best validation loss: {best_val_loss:.4f}")
    return model


# ===========================
# MAIN EXECUTION
# ===========================

def main():
    # Configuration
    MODEL_NAME = "bert-base-multilingual-cased"
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128

    # ===========================
    # Device configuration
    # ===========================
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_ids = [0, 1, 2, 3] if num_gpus >= 2 else [0]
        device = torch.device(f'cuda:{device_ids[0]}')
        print(f"✅ Using GPUs: {device_ids}")
    else:
        device = torch.device('cpu')
        device_ids = []
        print("⚠️ Using CPU")

    # ===========================
    # Tokenizer and Data
    # ===========================
    print("\n" + "="*50)
    print("Loading IndicBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
        texts, labels, test_size=0.2, random_state=0
    )

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    text_clf_model = IndicBERTClassifier(MODEL_NAME, num_classes=3).to(device)

    # ✅ Wrap model with DataParallel for multi-GPU training
    if len(device_ids) > 1:
        text_clf_model = torch.nn.DataParallel(text_clf_model, device_ids=device_ids)

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
        headlines, h_labels, test_size=0.2, random_state=0
    )

    train_hl_dataset = ClassificationDataset(train_hl, train_hl_labels, tokenizer, MAX_LENGTH)
    val_hl_dataset = ClassificationDataset(val_hl, val_hl_labels, tokenizer, MAX_LENGTH)

    train_hl_loader = DataLoader(train_hl_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_hl_loader = DataLoader(val_hl_dataset, batch_size=BATCH_SIZE)

    headline_clf_model = IndicBERTClassifier(MODEL_NAME, num_classes=4).to(device)
    if len(device_ids) > 1:
        headline_clf_model = torch.nn.DataParallel(headline_clf_model, device_ids=device_ids)

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
        premises, hypotheses, nli_labels, test_size=0.2, random_state=0
    )

    train_nli_dataset = NLIDataset(train_prem, train_hyp, train_nli_labels, tokenizer, MAX_LENGTH)
    val_nli_dataset = NLIDataset(val_prem, val_hyp, val_nli_labels, tokenizer, MAX_LENGTH)

    train_nli_loader = DataLoader(train_nli_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_nli_loader = DataLoader(val_nli_dataset, batch_size=BATCH_SIZE)

    nli_model = IndicBERTClassifier(MODEL_NAME, num_classes=3).to(device)
    if len(device_ids) > 1:
        nli_model = torch.nn.DataParallel(nli_model, device_ids=device_ids)

    nli_model = train_classification(nli_model, train_nli_loader, val_nli_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Task 4: Text Similarity
    # ===========================
    print("\n" + "="*50)
    print("TASK 4: Text Similarity (Regression)")
    print("="*50)

    text1_list = [item[0] for item in datasets['similarity']]
    text2_list = [item[1] for item in datasets['similarity']]
    sim_scores = [item[2] for item in datasets['similarity']]

    train_t1, val_t1, train_t2, val_t2, train_scores, val_scores = train_test_split(
        text1_list, text2_list, sim_scores, test_size=0.2, random_state=0
    )

    train_sim_dataset = SimilarityDataset(train_t1, train_t2, train_scores, tokenizer, MAX_LENGTH)
    val_sim_dataset = SimilarityDataset(val_t1, val_t2, val_scores, tokenizer, MAX_LENGTH)

    train_sim_loader = DataLoader(train_sim_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_sim_loader = DataLoader(val_sim_dataset, batch_size=BATCH_SIZE)

    similarity_model = IndicBERTSimilarity(MODEL_NAME).to(device)
    if len(device_ids) > 1:
        similarity_model = torch.nn.DataParallel(similarity_model, device_ids=device_ids)

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
        queries, positives, negatives, test_size=0.2, random_state=0
    )

    train_ret_dataset = RetrievalDataset(train_q, train_pos, train_neg, tokenizer, MAX_LENGTH)
    val_ret_dataset = RetrievalDataset(val_q, val_pos, val_neg, tokenizer, MAX_LENGTH)

    train_ret_loader = DataLoader(train_ret_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_ret_loader = DataLoader(val_ret_dataset, batch_size=BATCH_SIZE)

    retrieval_model = IndicBERTRetrieval(MODEL_NAME).to(device)
    if len(device_ids) > 1:
        retrieval_model = torch.nn.DataParallel(retrieval_model, device_ids=device_ids)

    retrieval_model = train_retrieval(retrieval_model, train_ret_loader, val_ret_loader, device, EPOCHS, LEARNING_RATE)

    # ===========================
    # Save Models
    # ===========================
    print("\n" + "="*50)
    print("Saving models...")

    # ✅ Save only the underlying model weights (important for DataParallel)
    def save_model(model, path):
        torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), path)

    save_model(text_clf_model, './mbert0/mBERT_text_classification_model.pt')
    save_model(headline_clf_model, './mbert0/mBERT_headline_classification_model.pt')
    save_model(nli_model, './mbert0/mBERT_nli_model.pt')
    save_model(similarity_model, './mbert0/mBERT_similarity_model.pt')
    save_model(retrieval_model, './mbert0/mBERT_retrieval_model.pt')

    print("All models saved successfully!")
    print("\n" + "="*50)
    print("Training completed for all tasks!")
    print("="*50)


if __name__ == "__main__":
    main()

"""

INFERENCE EXAMPLE:
------------------
# Text Classification
model = IndicBERTClassifier(MODEL_NAME, num_classes=3)
model.load_state_dict(torch.load('text_classification_model.pt'))
model.eval()

text = "Your Manipuri text here"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    predicted_class = torch.argmax(logits, dim=1).item()
print(f"Predicted class: {predicted_class}")

# Similarity
sim_model = IndicBERTSimilarity(MODEL_NAME)
sim_model.load_state_dict(torch.load('similarity_model.pt'))
sim_model.eval()

text1, text2 = "Text 1", "Text 2"
inputs = tokenizer(text1, text2, return_tensors='pt', padding=True, truncation=True, max_length=128)
with torch.no_grad():
    score = sim_model(inputs['input_ids'], inputs['attention_mask'])
print(f"Similarity score: {score.item()}")

# Retrieval
ret_model = IndicBERTRetrieval(MODEL_NAME)
ret_model.load_state_dict(torch.load('retrieval_model.pt'))
ret_model.eval()

query = "Your query"
docs = ["Document 1", "Document 2", "Document 3"]

query_inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
with torch.no_grad():
    query_emb = ret_model.encode(query_inputs['input_ids'], query_inputs['attention_mask'])

doc_embeddings = []
for doc in docs:
    doc_inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        doc_emb = ret_model.encode(doc_inputs['input_ids'], doc_inputs['attention_mask'])
        doc_embeddings.append(doc_emb)

# Compute cosine similarity
from torch.nn.functional import cosine_similarity
similarities = [cosine_similarity(query_emb, doc_emb).item() for doc_emb in doc_embeddings]
best_match_idx = similarities.index(max(similarities))
print(f"Best matching document: {docs[best_match_idx]}")
"""