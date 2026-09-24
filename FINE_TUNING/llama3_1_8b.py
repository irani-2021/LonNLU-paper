"""
Llama 3.1-8B Task-Specific Fine-tuning for Manipuri NLU Tasks with QLoRA
Separate models trained one at a time for each task
Tasks: Text Classification, Headline Classification, NLI, Text Similarity, Retrieval
Uses LoRA with 4-bit quantization for memory efficiency
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple
import random
import json
import pandas as pd
import gc
import warnings
import os

import os
os.environ['HF_HOME'] = '/media/csenitmz/DATA/lenin/pycharm-2025.2.3/cache/'

from huggingface_hub import login
login(token="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")

warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ===========================
# DATA LOADING
# ===========================

def read_data():
    """Load all datasets"""
    # Text Classification
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/content_classification/train_content_dataset.json", 'r',
              encoding='utf-8') as f:
        data = json.load(f)

    text_classification_data = [(item['text'], item['label']) for item in data]
    unique_labels = sorted(set(label for _, label in text_classification_data))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    text_classification_data = [(text, label2id[label]) for text, label in text_classification_data]

    print("✅ Label-to-ID mapping Text Classification:")
    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    # Headline Classification
    df_text = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/News_headlineclassification_train.csv")
    label2id = {label: idx for idx, label in enumerate(sorted(df_text['Category'].unique()))}
    df_text['label_id'] = df_text['Category'].map(label2id)
    headline_classification_data = list(zip(df_text['Headlines'], df_text['label_id']))

    print("✅ Label-to-ID mapping headline_classification_data:")
    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    # NLI
    df_nli = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/NLI/train_nli1.csv")
    label2id = {label: idx for idx, label in enumerate(sorted(df_nli['Label'].unique()))}
    df_nli['Label_id'] = df_nli['Label'].map(label2id)
    nli_data = list(zip(df_nli['Premise'], df_nli['Hypothesis'], df_nli['Label_id']))

    print("✅ Label-to-ID mapping NLI:")
    for label, idx in label2id.items():
        print(f"{label}: {idx}")

    # Text Similarity
    df_sim = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/sentence_similarity/semanticsimilarity_train.csv")
    similarity_data = list(zip(df_sim['SENTENCE1'], df_sim['SENTENCE2'], df_sim['SCORE']))

    # Retrieval
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/retrieval_train.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

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
    """Dataset for text and headline classification"""

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
    """Dataset for Natural Language Inference"""

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
    """Dataset for text similarity (regression)"""

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
    """Dataset for retrieval with triplet loss"""

    def __init__(self, queries, positives, negatives, tokenizer, max_length=128):
        self.queries = queries
        self.positives = positives
        self.negatives = negatives
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
        neg_input_ids = []
        neg_attention_masks = []
        for neg in negative_samples:
            neg_enc = self.tokenizer(
                neg,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            neg_input_ids.append(neg_enc['input_ids'].flatten())
            neg_attention_masks.append(neg_enc['attention_mask'].flatten())

        return {
            'query_input_ids': query_enc['input_ids'].flatten(),
            'query_attention_mask': query_enc['attention_mask'].flatten(),
            'pos_input_ids': pos_enc['input_ids'].flatten(),
            'pos_attention_mask': pos_enc['attention_mask'].flatten(),
            'neg_input_ids': torch.stack(neg_input_ids),
            'neg_attention_mask': torch.stack(neg_attention_masks)
        }


# ===========================
# MODEL CLASSES
# ===========================

class LlamaClassifier(nn.Module):
    """Llama model with classification head"""

    def __init__(self, model_name, num_classes, lora_config, device_ids):
        super().__init__()

        self.device_ids = device_ids

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Create device map
        device_map = self._create_device_map(device_ids)

        # Load Llama with quantization
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Prepare for k-bit training
        self.llama = prepare_model_for_kbit_training(self.llama)
        self.llama.gradient_checkpointing_enable()

        # Apply LoRA
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()

        # Classification head on last GPU
        self.last_gpu = f'cuda:{device_ids[-1]}'
        hidden_size = self.llama.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        ).to(self.last_gpu)

    def _create_device_map(self, device_ids):
        """Create device map for model parallelism"""
        num_layers = 32  # Llama 3.1-8B has 32 layers
        layers_per_gpu = num_layers // len(device_ids)

        device_map = {
            'model.embed_tokens': f'cuda:{device_ids[0]}',
            'model.norm': f'cuda:{device_ids[-1]}',
            'lm_head': f'cuda:{device_ids[-1]}'
        }

        for i in range(num_layers):
            gpu_idx = min(i // layers_per_gpu, len(device_ids) - 1)
            device_map[f'model.layers.{i}'] = f'cuda:{device_ids[gpu_idx]}'

        return device_map

    def get_hidden_states(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )

        hidden_states = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]

        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]

        return last_hidden_states.to(self.last_gpu)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.get_hidden_states(input_ids, attention_mask)
        return self.classifier(pooled_output)


class LlamaSimilarity(nn.Module):
    """Llama model for similarity (regression)"""

    def __init__(self, model_name, lora_config, device_ids):
        super().__init__()

        self.device_ids = device_ids

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        device_map = self._create_device_map(device_ids)

        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.llama = prepare_model_for_kbit_training(self.llama)
        self.llama.gradient_checkpointing_enable()
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()

        self.last_gpu = f'cuda:{device_ids[-1]}'
        hidden_size = self.llama.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        ).to(self.last_gpu)

    def _create_device_map(self, device_ids):
        num_layers = 32
        layers_per_gpu = num_layers // len(device_ids)

        device_map = {
            'model.embed_tokens': f'cuda:{device_ids[0]}',
            'model.norm': f'cuda:{device_ids[-1]}',
            'lm_head': f'cuda:{device_ids[-1]}'
        }

        for i in range(num_layers):
            gpu_idx = min(i // layers_per_gpu, len(device_ids) - 1)
            device_map[f'model.layers.{i}'] = f'cuda:{device_ids[gpu_idx]}'

        return device_map

    def get_hidden_states(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )

        hidden_states = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]

        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]

        return last_hidden_states.to(self.last_gpu)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.get_hidden_states(input_ids, attention_mask)
        return self.regressor(pooled_output)


class LlamaRetrieval(nn.Module):
    """Llama model for retrieval"""

    def __init__(self, model_name, lora_config, device_ids):
        super().__init__()

        self.device_ids = device_ids

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        device_map = self._create_device_map(device_ids)

        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.llama = prepare_model_for_kbit_training(self.llama)
        self.llama.gradient_checkpointing_enable()
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()

        self.last_gpu = f'cuda:{device_ids[-1]}'

    def _create_device_map(self, device_ids):
        num_layers = 32
        layers_per_gpu = num_layers // len(device_ids)

        device_map = {
            'model.embed_tokens': f'cuda:{device_ids[0]}',
            'model.norm': f'cuda:{device_ids[-1]}',
            'lm_head': f'cuda:{device_ids[-1]}'
        }

        for i in range(num_layers):
            gpu_idx = min(i // layers_per_gpu, len(device_ids) - 1)
            device_map[f'model.layers.{i}'] = f'cuda:{device_ids[gpu_idx]}'

        return device_map

    def get_hidden_states(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )

        hidden_states = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]

        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]

        return last_hidden_states.to(self.last_gpu)

    def encode(self, input_ids, attention_mask):
        return self.get_hidden_states(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask)


# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_classification(model, train_loader, val_loader, epochs=3, lr=2e-5):
    """Train classification model"""
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].to(model.last_gpu)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels'].to(model.last_gpu)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    return model


def train_similarity(model, train_loader, val_loader, epochs=3, lr=2e-5):
    """Train similarity model (regression)"""
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_mse = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].to(model.last_gpu)

            optimizer.zero_grad()
            scores = model(input_ids, attention_mask).squeeze()
            loss = criterion(scores, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels'].to(model.last_gpu)

                scores = model(input_ids, attention_mask).squeeze()
                loss = criterion(scores, labels)

                val_loss += loss.item()
                val_preds.extend(scores.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_mse = mean_squared_error(val_labels, val_preds)

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, MSE: {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"New best validation MSE: {best_val_mse:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    return model


def train_retrieval(model, train_loader, val_loader, epochs=3, lr=2e-5, margin=0.5):
    """Train retrieval model with triplet loss"""
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

        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in pbar:
            query_ids = batch['query_input_ids']
            query_mask = batch['query_attention_mask']
            pos_ids = batch['pos_input_ids']
            pos_mask = batch['pos_attention_mask']
            neg_ids = batch['neg_input_ids']
            neg_mask = batch['neg_attention_mask']

            optimizer.zero_grad()

            query_emb = model.encode(query_ids, query_mask)
            pos_emb = model.encode(pos_ids, pos_mask)

            num_negatives = neg_ids.size(1)
            total_loss = 0.0

            for i in range(num_negatives):
                neg_emb = model.encode(neg_ids[:, i, :], neg_mask[:, i, :])
                loss = criterion(query_emb, pos_emb, neg_emb)
                total_loss += loss

            total_loss = total_loss / num_negatives
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                query_ids = batch['query_input_ids']
                query_mask = batch['query_attention_mask']
                pos_ids = batch['pos_input_ids']
                pos_mask = batch['pos_attention_mask']
                neg_ids = batch['neg_input_ids']
                neg_mask = batch['neg_attention_mask']

                query_emb = model.encode(query_ids, query_mask)
                pos_emb = model.encode(pos_ids, pos_mask)

                num_negatives = neg_ids.size(1)
                total_loss = 0.0

                for i in range(num_negatives):
                    neg_emb = model.encode(neg_ids[:, i, :], neg_mask[:, i, :])
                    loss = criterion(query_emb, pos_emb, neg_emb)
                    total_loss += loss

                total_loss = total_loss / num_negatives
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  🏋️ Train Loss: {avg_train_loss:.4f}")
        print(f"  🧪 Val Loss:   {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✅ New best validation loss: {best_val_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ Training complete. Best validation loss: {best_val_loss:.4f}")
    return model


# ===========================
# MAIN EXECUTION
# ===========================

def main():
    # Configuration - Same as reference code
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128

    # ===========================
    # Device configuration
    # ===========================
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_ids = [0, 1, 2, 3] if num_gpus >= 4 else list(range(num_gpus))
        print(f"✅ Using GPUs: {device_ids}")
    else:
        print("⚠️ No CUDA GPUs available!")
        return

    # ===========================
    # LoRA Configuration
    # ===========================
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # ===========================
    # Tokenizer
    # ===========================
    print("\n" + "=" * 50)
    print("Loading Llama 3.1-8B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded")

    # ===========================
    # Load Data
    # ===========================
    print("\n" + "=" * 50)
    print("Reading data....")
    datasets = read_data()

    # ===========================
    # Task 1: Text Classification (3 classes)
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 1: Text Classification (3 classes)")
    print("=" * 50)

    texts = [item[0] for item in datasets['text_classification']]
    labels = [item[1] for item in datasets['text_classification']]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("Initializing Llama 3.1-8B with QLoRA for text classification...")
    text_clf_model = LlamaClassifier(MODEL_NAME, num_classes=3, lora_config=lora_config, device_ids=device_ids)

    print("\nTraining text classification model...")
    text_clf_model = train_classification(text_clf_model, train_loader, val_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving text classification model...")
    text_clf_model.llama.save_pretrained('./llama_text_classification_lora')
    torch.save(text_clf_model.classifier.state_dict(), './llama_text_classification_head.pt')
    print("✓ Text classification model saved")

    # Clean up
    del text_clf_model, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================
    # Task 2: Headline Classification (4 classes)
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 2: Headline Classification (4 classes)")
    print("=" * 50)

    headlines = [item[0] for item in datasets['headline_classification']]
    h_labels = [item[1] for item in datasets['headline_classification']]

    train_hl, val_hl, train_hl_labels, val_hl_labels = train_test_split(
        headlines, h_labels, test_size=0.2, random_state=42
    )

    train_hl_dataset = ClassificationDataset(train_hl, train_hl_labels, tokenizer, MAX_LENGTH)
    val_hl_dataset = ClassificationDataset(val_hl, val_hl_labels, tokenizer, MAX_LENGTH)

    train_hl_loader = DataLoader(train_hl_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_hl_loader = DataLoader(val_hl_dataset, batch_size=BATCH_SIZE)

    print("Initializing Llama 3.1-8B with QLoRA for headline classification...")
    headline_clf_model = LlamaClassifier(MODEL_NAME, num_classes=4, lora_config=lora_config, device_ids=device_ids)

    print("\nTraining headline classification model...")
    headline_clf_model = train_classification(headline_clf_model, train_hl_loader, val_hl_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving headline classification model...")
    headline_clf_model.llama.save_pretrained('./llama_headline_classification_lora')
    torch.save(headline_clf_model.classifier.state_dict(), './llama_headline_classification_head.pt')
    print("✓ Headline classification model saved")

    # Clean up
    del headline_clf_model, train_hl_dataset, val_hl_dataset, train_hl_loader, val_hl_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================
    # Task 3: NLI (3 classes)
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 3: Natural Language Inference (NLI)")
    print("=" * 50)

    premises = [item[0] for item in datasets['nli']]
    hypotheses = [item[1] for item in datasets['nli']]
    nli_labels = [item[2] for item in datasets['nli']]

    train_prem, val_prem, train_hyp, val_hyp, train_nli_labels, val_nli_labels = train_test_split(
        premises, hypotheses, nli_labels, test_size=0.2, random_state=42
    )

    train_nli_dataset = NLIDataset(train_prem, train_hyp, train_nli_labels, tokenizer, MAX_LENGTH)
    val_nli_dataset = NLIDataset(val_prem, val_hyp, val_nli_labels, tokenizer, MAX_LENGTH)

    train_nli_loader = DataLoader(train_nli_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_nli_loader = DataLoader(val_nli_dataset, batch_size=BATCH_SIZE)

    print("Initializing Llama 3.1-8B with QLoRA for NLI...")
    nli_model = LlamaClassifier(MODEL_NAME, num_classes=3, lora_config=lora_config, device_ids=device_ids)

    print("\nTraining NLI model...")
    nli_model = train_classification(nli_model, train_nli_loader, val_nli_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving NLI model...")
    nli_model.llama.save_pretrained('./llama_nli_lora')
    torch.save(nli_model.classifier.state_dict(), './llama_nli_head.pt')
    print("✓ NLI model saved")

    # Clean up
    del nli_model, train_nli_dataset, val_nli_dataset, train_nli_loader, val_nli_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================
    # Task 4: Text Similarity
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 4: Text Similarity (Regression)")
    print("=" * 50)

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

    print("Initializing Llama 3.1-8B with QLoRA for text similarity...")
    similarity_model = LlamaSimilarity(MODEL_NAME, lora_config=lora_config, device_ids=device_ids)

    print("\nTraining similarity model...")
    similarity_model = train_similarity(similarity_model, train_sim_loader, val_sim_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving similarity model...")
    similarity_model.llama.save_pretrained('./llama_similarity_lora')
    torch.save(similarity_model.regressor.state_dict(), './llama_similarity_head.pt')
    print("✓ Similarity model saved")

    # Clean up
    del similarity_model, train_sim_dataset, val_sim_dataset, train_sim_loader, val_sim_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================
    # Task 5: Retrieval
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 5: Retrieval (Query, Positive, Negatives)")
    print("=" * 50)

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

    print("Initializing Llama 3.1-8B with QLoRA for retrieval...")
    retrieval_model = LlamaRetrieval(MODEL_NAME, lora_config=lora_config, device_ids=device_ids)

    print("\nTraining retrieval model...")
    retrieval_model = train_retrieval(retrieval_model, train_ret_loader, val_ret_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving retrieval model...")
    retrieval_model.llama.save_pretrained('./llama_retrieval_lora')
    print("✓ Retrieval model saved")

    # Clean up
    del retrieval_model, train_ret_dataset, val_ret_dataset, train_ret_loader, val_ret_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================
    # Final Summary
    # ===========================
    print("\n" + "=" * 50)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 50)
    print("\nSaved models:")
    print("1. Text Classification:")
    print("   - LoRA: ./llama_text_classification_lora/")
    print("   - Head: ./llama_text_classification_head.pt")
    print("\n2. Headline Classification:")
    print("   - LoRA: ./llama_headline_classification_lora/")
    print("   - Head: ./llama_headline_classification_head.pt")
    print("\n3. NLI:")
    print("   - LoRA: ./llama_nli_lora/")
    print("   - Head: ./llama_nli_head.pt")
    print("\n4. Text Similarity:")
    print("   - LoRA: ./llama_similarity_lora/")
    print("   - Head: ./llama_similarity_head.pt")
    print("\n5. Retrieval:")
    print("   - LoRA: ./llama_retrieval_lora/")
    print("=" * 50)
    print("\nTraining Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Quantization: 4-bit (NF4)")
    print(f"  LoRA Rank: {lora_config.r}")
    print(f"  LoRA Alpha: {lora_config.lora_alpha}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  GPUs Used: {device_ids}")
    print("=" * 50)


# ===========================
# INFERENCE EXAMPLE
# ===========================

def load_model_for_inference(task_name, model_name="meta-llama/Llama-3.1-8B", device_ids=[0, 1, 2, 3]):
    """
    Load a trained model for inference

    Args:
        task_name: One of ['text_classification', 'headline_classification', 'nli', 'similarity', 'retrieval']
        model_name: Base Llama model name
        device_ids: GPU devices to use

    Returns:
        Loaded model and tokenizer
    """
    from peft import PeftModel

    print(f"\nLoading {task_name} model for inference...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Create device map
    num_layers = 32
    layers_per_gpu = num_layers // len(device_ids)
    device_map = {
        'model.embed_tokens': f'cuda:{device_ids[0]}',
        'model.norm': f'cuda:{device_ids[-1]}',
        'lm_head': f'cuda:{device_ids[-1]}'
    }
    for i in range(num_layers):
        gpu_idx = min(i // layers_per_gpu, len(device_ids) - 1)
        device_map[f'model.layers.{i}'] = f'cuda:{device_ids[gpu_idx]}'

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA weights
    if task_name == 'text_classification':
        model = PeftModel.from_pretrained(base_model, './llama_text_classification_lora')
        # Load classifier head
        hidden_size = model.config.hidden_size
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)
        )
        classifier.load_state_dict(torch.load('./llama_text_classification_head.pt'))
        classifier = classifier.to(f'cuda:{device_ids[-1]}')

    elif task_name == 'headline_classification':
        model = PeftModel.from_pretrained(base_model, './llama_headline_classification_lora')
        hidden_size = model.config.hidden_size
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 4)
        )
        classifier.load_state_dict(torch.load('./llama_headline_classification_head.pt'))
        classifier = classifier.to(f'cuda:{device_ids[-1]}')

    elif task_name == 'nli':
        model = PeftModel.from_pretrained(base_model, './llama_nli_lora')
        hidden_size = model.config.hidden_size
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)
        )
        classifier.load_state_dict(torch.load('./llama_nli_head.pt'))
        classifier = classifier.to(f'cuda:{device_ids[-1]}')

    elif task_name == 'similarity':
        model = PeftModel.from_pretrained(base_model, './llama_similarity_lora')
        hidden_size = model.config.hidden_size
        classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        classifier.load_state_dict(torch.load('./llama_similarity_head.pt'))
        classifier = classifier.to(f'cuda:{device_ids[-1]}')

    elif task_name == 'retrieval':
        model = PeftModel.from_pretrained(base_model, './llama_retrieval_lora')
        classifier = None

    else:
        raise ValueError(f"Unknown task: {task_name}")

    model.eval()
    print(f"✓ {task_name} model loaded successfully!")

    return model, tokenizer, classifier


if __name__ == "__main__":
    main()
