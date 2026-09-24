"""
Gemma 3-4B Task-Specific Fine-tuning (QLoRA + Multi-GPU)
-------------------------------------------------
Train separate models for: NLI, Text Classification, Headline Classification, Sentence Similarity, Retrieval
"""

import os
os.environ['HF_HOME'] = '/media/csenitmz/DATA/lenin/pycharm-2025.2.3/cache/'

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
import pandas as pd
import random
import json
import gc
import warnings
from huggingface_hub import login

warnings.filterwarnings("ignore")

# ✅ Login to Hugging Face
login(token="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")

# =============== UTILITIES ===============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =============== LOAD DATA ===============
def read_data():
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/content_classification/train_content_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_classification_data = [(d['text'], d['label']) for d in data]
    label2id = {label: i for i, label in enumerate(sorted(set(l for _, l in text_classification_data)))}
    text_classification_data = [(t, label2id[l]) for t, l in text_classification_data]
    print("✅ Text Classification Labels:", label2id)

    df_h = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/News_headlineclassification_train.csv")
    label2id = {l: i for i, l in enumerate(sorted(df_h['Category'].unique()))}
    df_h['label_id'] = df_h['Category'].map(label2id)
    headline_classification_data = list(zip(df_h['Headlines'], df_h['label_id']))
    print("✅ Headline Labels:", label2id)

    df_nli = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/NLI/train_nli1.csv")
    label2id = {l: i for i, l in enumerate(sorted(df_nli['Label'].unique()))}
    df_nli['Label_id'] = df_nli['Label'].map(label2id)
    nli_data = list(zip(df_nli['Premise'], df_nli['Hypothesis'], df_nli['Label_id']))
    print("✅ NLI Labels:", label2id)

    df_sim = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/sentence_similarity/semanticsimilarity_train.csv")
    similarity_data = list(zip(df_sim['SENTENCE1'], df_sim['SENTENCE2'], df_sim['SCORE']))

    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/retrieval_train.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    retrieval_data = [(d['query'], d['positive'], [d['negatives']['random'], d['negatives']['tfidf'], d['negatives']['bm25']]) for d in data]

    return {
        'text_classification': text_classification_data,
        'headline_classification': headline_classification_data,
        'nli': nli_data,
        'similarity': similarity_data,
        'retrieval': retrieval_data
    }


# =============== DATASETS ===============
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
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
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
        encoding = self.tokenizer(premise, hypothesis, add_special_tokens=True, max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
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
        encoding = self.tokenizer(text1, text2, add_special_tokens=True, max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }


class RetrievalDataset(Dataset):
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

        query_enc = self.tokenizer(query, add_special_tokens=True, max_length=self.max_length,
                                   padding='max_length', truncation=True, return_tensors='pt')
        pos_enc = self.tokenizer(positive, add_special_tokens=True, max_length=self.max_length,
                                padding='max_length', truncation=True, return_tensors='pt')

        neg_input_ids = []
        neg_attention_masks = []
        for neg in negative_samples:
            neg_enc = self.tokenizer(neg, add_special_tokens=True, max_length=self.max_length,
                                    padding='max_length', truncation=True, return_tensors='pt')
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


# =============== MODELS ===============
class GemmaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, lora_config, device_ids):
        super().__init__()
        self.device_ids = device_ids

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

        self.gemma = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        )

        self.gemma = prepare_model_for_kbit_training(self.gemma)
        self.gemma.gradient_checkpointing_enable()
        self.gemma = get_peft_model(self.gemma, lora_config)
        self.gemma.print_trainable_parameters()

        # Determine hidden size
        hidden_size = self._get_hidden_size()
        self.last_gpu = f'cuda:{device_ids[-1]}'
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, num_classes)).to(self.last_gpu)

    def _get_hidden_size(self):
        config = self.gemma.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        elif hasattr(config, "hidden_dim"):
            return int(config.hidden_dim)
        else:
            # Probe with dummy input
            probe_ids = torch.ones(1, 2, dtype=torch.long)
            probe_mask = torch.ones_like(probe_ids)
            with torch.no_grad():
                out = self.gemma(probe_ids, attention_mask=probe_mask,
                                output_hidden_states=True, return_dict=True, use_cache=False)
            return int(out.hidden_states[-1].size(-1))

    def get_hidden(self, ids, mask):
        out = self.gemma(ids, attention_mask=mask, output_hidden_states=True, return_dict=True, use_cache=False)
        hs = out.hidden_states[-1]
        idx = mask.sum(dim=1) - 1
        return hs[torch.arange(hs.size(0)), idx].to(self.last_gpu)

    def forward(self, ids, mask):
        return self.classifier(self.get_hidden(ids, mask))


class GemmaSimilarity(nn.Module):
    def __init__(self, model_name, lora_config, device_ids):
        super().__init__()
        self.device_ids = device_ids

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

        self.gemma = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        )

        self.gemma = prepare_model_for_kbit_training(self.gemma)
        self.gemma.gradient_checkpointing_enable()
        self.gemma = get_peft_model(self.gemma, lora_config)
        self.gemma.print_trainable_parameters()

        hidden_size = self._get_hidden_size()
        self.last_gpu = f'cuda:{device_ids[-1]}'
        self.regressor = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 1)).to(self.last_gpu)

    def _get_hidden_size(self):
        config = self.gemma.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        elif hasattr(config, "hidden_dim"):
            return int(config.hidden_dim)
        else:
            probe_ids = torch.ones(1, 2, dtype=torch.long)
            probe_mask = torch.ones_like(probe_ids)
            with torch.no_grad():
                out = self.gemma(probe_ids, attention_mask=probe_mask,
                                output_hidden_states=True, return_dict=True, use_cache=False)
            return int(out.hidden_states[-1].size(-1))

    def get_hidden(self, ids, mask):
        out = self.gemma(ids, attention_mask=mask, output_hidden_states=True, return_dict=True, use_cache=False)
        hs = out.hidden_states[-1]
        idx = mask.sum(dim=1) - 1
        return hs[torch.arange(hs.size(0)), idx].to(self.last_gpu)

    def forward(self, ids, mask):
        return self.regressor(self.get_hidden(ids, mask))


class GemmaRetrieval(nn.Module):
    def __init__(self, model_name, lora_config, device_ids):
        super().__init__()
        self.device_ids = device_ids

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

        self.gemma = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        )

        self.gemma = prepare_model_for_kbit_training(self.gemma)
        self.gemma.gradient_checkpointing_enable()
        self.gemma = get_peft_model(self.gemma, lora_config)
        self.gemma.print_trainable_parameters()

        self.last_gpu = f'cuda:{device_ids[-1]}'

    def _get_hidden_size(self):
        config = self.gemma.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        elif hasattr(config, "hidden_dim"):
            return int(config.hidden_dim)
        else:
            probe_ids = torch.ones(1, 2, dtype=torch.long)
            probe_mask = torch.ones_like(probe_ids)
            with torch.no_grad():
                out = self.gemma(probe_ids, attention_mask=probe_mask,
                                output_hidden_states=True, return_dict=True, use_cache=False)
            return int(out.hidden_states[-1].size(-1))

    def get_hidden(self, ids, mask):
        out = self.gemma(ids, attention_mask=mask, output_hidden_states=True, return_dict=True, use_cache=False)
        hs = out.hidden_states[-1]
        idx = mask.sum(dim=1) - 1
        return hs[torch.arange(hs.size(0)), idx].to(self.last_gpu)

    def encode(self, ids, mask):
        return self.get_hidden(ids, mask)

    def forward(self, ids, mask):
        return self.encode(ids, mask)


# =============== TRAINING FUNCTIONS ===============
def train_classification(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
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

        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    return model


def train_similarity(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )

    best_val_mse = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels'].to(model.last_gpu)

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

        torch.cuda.empty_cache()
        gc.collect()

    return model


def train_retrieval(model, train_loader, val_loader, epochs=3, lr=2e-5, margin=0.5):
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")

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
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✅ New best validation loss: {best_val_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ Training complete. Best validation loss: {best_val_loss:.4f}")
    return model


# =============== MAIN ===============
def main():
    MODEL_NAME = "google/gemma-3-4b-pt"
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128

    if not torch.cuda.is_available():
        print("❌ No GPU found!")
        return

    n_gpus = torch.cuda.device_count()
    device_ids = list(range(min(n_gpus, 4)))
    print(f"✅ Using GPUs: {device_ids}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
    )

    print("\n" + "="*50)
    print("Reading data...")
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

    print("Initializing Gemma 3-4B with QLoRA for text classification...")
    text_clf_model = GemmaClassifier(MODEL_NAME, num_classes=3, lora_config=lora_cfg, device_ids=device_ids)

    print("\nTraining text classification model...")
    text_clf_model = train_classification(text_clf_model, train_loader, val_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving text classification model...")
    text_clf_model.gemma.save_pretrained('./gemma_text_classification_lora')
    torch.save(text_clf_model.classifier.state_dict(), './gemma_text_classification_head.pt')
    print("✓ Text classification model saved")

    del text_clf_model, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

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

    print("Initializing Gemma 3-4B with QLoRA for headline classification...")
    headline_clf_model = GemmaClassifier(MODEL_NAME, num_classes=4, lora_config=lora_cfg, device_ids=device_ids)

    print("\nTraining headline classification model...")
    headline_clf_model = train_classification(headline_clf_model, train_hl_loader, val_hl_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving headline classification model...")
    headline_clf_model.gemma.save_pretrained('./gemma_headline_classification_lora')
    torch.save(headline_clf_model.classifier.state_dict(), './gemma_headline_classification_head.pt')
    print("✓ Headline classification model saved")

    del headline_clf_model, train_hl_dataset, val_hl_dataset, train_hl_loader, val_hl_loader
    torch.cuda.empty_cache()
    gc.collect()

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

    train_nli_dataset = NLIDataset(train_prem, train_hyp, train_nli_labels, tokenizer, MAX_LENGTH)
    val_nli_dataset = NLIDataset(val_prem, val_hyp, val_nli_labels, tokenizer, MAX_LENGTH)
    train_nli_loader = DataLoader(train_nli_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_nli_loader = DataLoader(val_nli_dataset, batch_size=BATCH_SIZE)

    print("Initializing Gemma 3-4B with QLoRA for NLI...")
    nli_model = GemmaClassifier(MODEL_NAME, num_classes=3, lora_config=lora_cfg, device_ids=device_ids)

    print("\nTraining NLI model...")
    nli_model = train_classification(nli_model, train_nli_loader, val_nli_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving NLI model...")
    nli_model.gemma.save_pretrained('./gemma_nli_lora')
    torch.save(nli_model.classifier.state_dict(), './gemma_nli_head.pt')
    print("✓ NLI model saved")

    del nli_model, train_nli_dataset, val_nli_dataset, train_nli_loader, val_nli_loader
    torch.cuda.empty_cache()
    gc.collect()

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
        text1_list, text2_list, sim_scores, test_size=0.2, random_state=42
    )

    train_sim_dataset = SimilarityDataset(train_t1, train_t2, train_scores, tokenizer, MAX_LENGTH)
    val_sim_dataset = SimilarityDataset(val_t1, val_t2, val_scores, tokenizer, MAX_LENGTH)
    train_sim_loader = DataLoader(train_sim_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_sim_loader = DataLoader(val_sim_dataset, batch_size=BATCH_SIZE)

    print("Initializing Gemma 3-4B with QLoRA for text similarity...")
    similarity_model = GemmaSimilarity(MODEL_NAME, lora_config=lora_cfg, device_ids=device_ids)

    print("\nTraining similarity model...")
    similarity_model = train_similarity(similarity_model, train_sim_loader, val_sim_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving similarity model...")
    similarity_model.gemma.save_pretrained('./gemma_similarity_lora')
    torch.save(similarity_model.regressor.state_dict(), './gemma_similarity_head.pt')
    print("✓ Similarity model saved")

    del similarity_model, train_sim_dataset, val_sim_dataset, train_sim_loader, val_sim_loader
    torch.cuda.empty_cache()
    gc.collect()

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

    print("Initializing Gemma 3-4B with QLoRA for retrieval...")
    retrieval_model = GemmaRetrieval(MODEL_NAME, lora_config=lora_cfg, device_ids=device_ids)

    print("\nTraining retrieval model...")
    retrieval_model = train_retrieval(retrieval_model, train_ret_loader, val_ret_loader, EPOCHS, LEARNING_RATE)

    print("\nSaving retrieval model...")
    retrieval_model.gemma.save_pretrained('./gemma_retrieval_lora')
    print("✓ Retrieval model saved")

    del retrieval_model, train_ret_dataset, val_ret_dataset, train_ret_loader, val_ret_loader
    torch.cuda.empty_cache()
    gc.collect()

    # ===========================
    # Final Summary
    # ===========================
    print("\n" + "="*50)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("="*50)
    print("\nSaved models:")
    print("1. Text Classification:")
    print("   - LoRA: ./gemma_text_classification_lora/")
    print("   - Head: ./gemma_text_classification_head.pt")
    print("\n2. Headline Classification:")
    print("   - LoRA: ./gemma_headline_classification_lora/")
    print("   - Head: ./gemma_headline_classification_head.pt")
    print("\n3. NLI:")
    print("   - LoRA: ./gemma_nli_lora/")
    print("   - Head: ./gemma_nli_head.pt")
    print("\n4. Text Similarity:")
    print("   - LoRA: ./gemma_similarity_lora/")
    print("   - Head: ./gemma_similarity_head.pt")
    print("\n5. Retrieval:")
    print("   - LoRA: ./gemma_retrieval_lora/")
    print("="*50)
    print("\nTraining Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Quantization: 4-bit (NF4)")
    print(f"  LoRA Rank: {lora_cfg.r}")
    print(f"  LoRA Alpha: {lora_cfg.lora_alpha}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  GPUs Used: {device_ids}")
    print("="*50)


if __name__ == "__main__":
    main()