"""
LLM Model Evaluation for Manipuri NLU Tasks
Evaluates trained models on test data with comprehensive metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import json
import pandas as pd
from typing import Dict, List, Tuple
import torch.nn.functional as F


# ===========================
# LOAD TEST DATA
# ===========================

def read_test_data():
    """Load all test datasets"""

    # Text Classification
    with open(r"E:\Irani\set_up\content_classification\test_content_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get label mapping from training data
    with open(r"E:\Irani\set_up\content_classification\train_content_dataset.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    unique_labels = sorted(set(item['label'] for item in train_data))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    text_classification_data = [(item['text'], label2id[item['label']]) for item in data]

    print("✅ Text Classification test samples:", len(text_classification_data))

    # Headline Classification
    df_test = pd.read_csv(r"E:\Irani\set_up\News_headlineclassification_test.csv")
    df_train = pd.read_csv(r"E:\Irani\set_up\News_headlineclassification_train.csv")
    label2id_h = {label: idx for idx, label in enumerate(sorted(df_train['Category'].unique()))}
    df_test['label_id'] = df_test['Category'].map(label2id_h)
    headline_classification_data = list(zip(df_test['Headlines'], df_test['label_id']))

    print("✅ Headline Classification test samples:", len(headline_classification_data))

    # NLI
    df_nli_test = pd.read_csv(r"E:\Irani\set_up\NLI\test_nli1.csv")
    df_nli_train = pd.read_csv(r"E:\Irani\set_up\NLI\train_nli1.csv")
    label2id_n = {label: idx for idx, label in enumerate(sorted(df_nli_train['Label'].unique()))}
    df_nli_test['Label_id'] = df_nli_test['Label'].map(label2id_n)
    nli_data = list(zip(df_nli_test['Premise'], df_nli_test['Hypothesis'], df_nli_test['Label_id']))

    print("✅ NLI test samples:", len(nli_data))

    # Text Similarity
    df_sim = pd.read_csv(r"E:\Irani\set_up\sentence_similarity\semanticsimilarity_test.csv")
    similarity_data = list(zip(df_sim['SENTENCE1'], df_sim['SENTENCE2'], df_sim['SCORE']))

    print("✅ Similarity test samples:", len(similarity_data))

    # Retrieval
    with open(r"E:\Irani\set_up\retrieval_test.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    retrieval_data = [
        (
            item['query'],
            item['positive'],
            [item['negatives']['random'], item['negatives']['tfidf'], item['negatives']['bm25']]
        )
        for item in data
    ]

    print("✅ Retrieval test samples:", len(retrieval_data))

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

class LLMClassifier(nn.Module):
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


class LLMSimilarity(nn.Module):
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


class LLMRetrieval(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def encode(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)


# ===========================
# EVALUATION FUNCTIONS
# ===========================

def evaluate_classification(model, test_loader, device):
    """Evaluate classification model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_similarity(model, test_loader, device):
    """Evaluate similarity model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            scores = model(input_ids, attention_mask).squeeze()

            all_preds.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)

    # Correlation metrics
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    spearman_corr, _ = spearmanr(all_labels, all_preds)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_retrieval(model, test_loader, device):
    """Evaluate retrieval model with ranking metrics"""
    model.eval()

    ranks = []
    mrr_scores = []
    recall_at_1 = 0
    # recall_at_3 = 0
    # recall_at_5 = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            query_ids = batch['query_input_ids'].to(device)
            query_mask = batch['query_attention_mask'].to(device)
            pos_ids = batch['pos_input_ids'].to(device)
            pos_mask = batch['pos_attention_mask'].to(device)
            neg_ids = batch['neg_input_ids'].to(device)
            neg_mask = batch['neg_attention_mask'].to(device)

            batch_size = query_ids.size(0)
            num_negatives = neg_ids.size(1)

            # Encode query and positive
            query_emb = model.encode(query_ids, query_mask)
            pos_emb = model.encode(pos_ids, pos_mask)

            # Calculate similarity scores for each sample in batch
            for i in range(batch_size):
                # Get embeddings for this sample
                q_emb = query_emb[i].unsqueeze(0)  # (1, hidden_size)
                p_emb = pos_emb[i].unsqueeze(0)

                # Calculate positive similarity
                pos_sim = F.cosine_similarity(q_emb, p_emb).item()

                # Calculate negative similarities
                neg_sims = []
                for j in range(num_negatives):
                    n_emb = model.encode(
                        neg_ids[i, j, :].unsqueeze(0),
                        neg_mask[i, j, :].unsqueeze(0)
                    )
                    neg_sim = F.cosine_similarity(q_emb, n_emb).item()
                    neg_sims.append(neg_sim)

                # Combine all similarities (positive first, then negatives)
                all_sims = [pos_sim] + neg_sims

                # Rank (positive should be at index 0)
                sorted_indices = np.argsort(all_sims)[::-1]  # Descending order
                rank = np.where(sorted_indices == 0)[0][0] + 1  # 1-indexed rank

                ranks.append(rank)
                mrr_scores.append(1.0 / rank)

                # Recall@K
                if rank <= 1:
                    recall_at_1 += 1
                # if rank <= 3:
                #     recall_at_3 += 1
                # if rank <= 5:
                #     recall_at_5 += 1

    total_samples = len(ranks)

    return {
        'mrr': np.mean(mrr_scores),
        'recall@1': recall_at_1 / total_samples,
        # 'recall@3': recall_at_3 / total_samples,
        # 'recall@5': recall_at_5 / total_samples,
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks)
    }


# ===========================
# MAIN EVALUATION
# ===========================

def main():
    # Configuration
    # print("MuRIL")
    # MODEL_NAME = "google/muril-base-cased"
    print("Custom RoBERTA Evaluation")
    #MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
    MANIPURI_MODEL_PATH = r"E:\Irani\roberta_pretraining\checkpoint-2334250"
    MANIPURI_TOKENIZER_PATH = r"E:\Irani\roberta_pretraining\custom_roberta_tokenizer"
    BATCH_SIZE = 16
    MAX_LENGTH = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("\n" + "=" * 50)
    print("Loading LLM tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MANIPURI_TOKENIZER_PATH)

    # Load test data
    print("\n" + "=" * 50)
    print("Loading test data...")
    test_datasets = read_test_data()

    # ===========================
    # Task 1: Text Classification
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 1: Text Classification Evaluation")
    print("=" * 50)

    texts = [item[0] for item in test_datasets['text_classification']]
    labels = [item[1] for item in test_datasets['text_classification']]

    test_dataset = ClassificationDataset(texts, labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    text_clf_model = LLMClassifier(MANIPURI_MODEL_PATH, num_classes=3)
    # text_clf_model.load_state_dict(torch.load('.venv/MuRIL_full/text_classification_model.pt',
    #                                           map_location=device))
    text_clf_model.load_state_dict(torch.load(r"E:\Irani\manipuri_Roberta_models\text_classification_model.pt",
                                              map_location=device))
    text_clf_model.to(device)

    # Evaluate
    results_text = evaluate_classification(text_clf_model, test_loader, device)

    print(f"\nResults:")
    print(f"  Accuracy:  {results_text['accuracy']:.4f}")
    print(f"  Precision: {results_text['precision']:.4f}")
    print(f"  Recall:    {results_text['recall']:.4f}")
    print(f"  F1 Score:  {results_text['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results_text['confusion_matrix'])

    # ===========================
    # Task 2: Headline Classification
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 2: Headline Classification Evaluation")
    print("=" * 50)

    headlines = [item[0] for item in test_datasets['headline_classification']]
    h_labels = [item[1] for item in test_datasets['headline_classification']]

    test_hl_dataset = ClassificationDataset(headlines, h_labels, tokenizer, MAX_LENGTH)
    test_hl_loader = DataLoader(test_hl_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    headline_clf_model = LLMClassifier(MANIPURI_MODEL_PATH, num_classes=4)
    # headline_clf_model.load_state_dict(torch.load('.venv/MuRIL_full/headline_classification_model.pt',
    #                                               map_location=device))
    headline_clf_model.load_state_dict(torch.load(r"E:\Irani\manipuri_Roberta_models\headline_classification_model.pt",
                                                  map_location=device))
    headline_clf_model.to(device)

    # Evaluate
    results_headline = evaluate_classification(headline_clf_model, test_hl_loader, device)

    print(f"\nResults:")
    print(f"  Accuracy:  {results_headline['accuracy']:.4f}")
    print(f"  Precision: {results_headline['precision']:.4f}")
    print(f"  Recall:    {results_headline['recall']:.4f}")
    print(f"  F1 Score:  {results_headline['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results_headline['confusion_matrix'])

    # ===========================
    # Task 3: NLI
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 3: Natural Language Inference Evaluation")
    print("=" * 50)

    premises = [item[0] for item in test_datasets['nli']]
    hypotheses = [item[1] for item in test_datasets['nli']]
    nli_labels = [item[2] for item in test_datasets['nli']]

    test_nli_dataset = NLIDataset(premises, hypotheses, nli_labels, tokenizer, MAX_LENGTH)
    test_nli_loader = DataLoader(test_nli_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    nli_model = LLMClassifier(MANIPURI_MODEL_PATH, num_classes=3)
    # nli_model.load_state_dict(torch.load('.venv/MuRIL_full/nli_model.pt', map_location=device))
    nli_model.load_state_dict(torch.load(r"E:\Irani\manipuri_Roberta_models\nli_model.pt", map_location=device))
    nli_model.to(device)

    # Evaluate
    results_nli = evaluate_classification(nli_model, test_nli_loader, device)

    print(f"\nResults:")
    print(f"  Accuracy:  {results_nli['accuracy']:.4f}")
    print(f"  Precision: {results_nli['precision']:.4f}")
    print(f"  Recall:    {results_nli['recall']:.4f}")
    print(f"  F1 Score:  {results_nli['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results_nli['confusion_matrix'])

    # ===========================
    # Task 4: Text Similarity
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 4: Text Similarity Evaluation")
    print("=" * 50)

    text1_list = [item[0] for item in test_datasets['similarity']]
    text2_list = [item[1] for item in test_datasets['similarity']]
    sim_scores = [item[2] for item in test_datasets['similarity']]

    test_sim_dataset = SimilarityDataset(text1_list, text2_list, sim_scores, tokenizer, MAX_LENGTH)
    test_sim_loader = DataLoader(test_sim_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    similarity_model = LLMSimilarity(MANIPURI_MODEL_PATH)
    # similarity_model.load_state_dict(torch.load('.venv/MuRIL_full/similarity_model.pt',
    #                                             map_location=device))
    similarity_model.load_state_dict(torch.load(r"E:\Irani\manipuri_Roberta_models\similarity_model.pt",
                                                map_location=device))
    similarity_model.to(device)

    # Evaluate
    results_sim = evaluate_similarity(similarity_model, test_sim_loader, device)

    print(f"\nResults:")
    print(f"  MSE:              {results_sim['mse']:.4f}")
    print(f"  MAE:              {results_sim['mae']:.4f}")
    print(f"  RMSE:             {results_sim['rmse']:.4f}")
    print(f"  Pearson Corr:     {results_sim['pearson']:.4f}")
    print(f"  Spearman Corr:    {results_sim['spearman']:.4f}")

    # ===========================
    # Task 5: Retrieval
    # ===========================
    print("\n" + "=" * 50)
    print("TASK 5: Retrieval Evaluation")
    print("=" * 50)

    queries = [item[0] for item in test_datasets['retrieval']]
    positives = [item[1] for item in test_datasets['retrieval']]
    negatives = [item[2] for item in test_datasets['retrieval']]

    test_ret_dataset = RetrievalDataset(queries, positives, negatives, tokenizer, MAX_LENGTH)
    test_ret_loader = DataLoader(test_ret_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    retrieval_model = LLMRetrieval(MANIPURI_MODEL_PATH)
    # retrieval_model.load_state_dict(torch.load('.venv/MuRIL_full/retrieval_model.pt',
    #                                            map_location=device))
    retrieval_model.load_state_dict(torch.load(r"E:\Irani\manipuri_Roberta_models\retrieval_model.pt",
                                               map_location=device))
    retrieval_model.to(device)

    # Evaluate
    results_ret = evaluate_retrieval(retrieval_model, test_ret_loader, device)

    print(f"\nResults:")
    print(f"  MRR (Mean Reciprocal Rank): {results_ret['mrr']:.4f}")
    print(f"  Recall@1:                   {results_ret['recall@1']:.4f}")
    # print(f"  Recall@3:                   {results_ret['recall@3']:.4f}")
    # print(f"  Recall@5:                   {results_ret['recall@5']:.4f}")
    print(f"  Mean Rank:                  {results_ret['mean_rank']:.2f}")
    print(f"  Median Rank:                {results_ret['median_rank']:.2f}")

    # ===========================
    # Summary
    # ===========================
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"\n{'Task':<30} {'Primary Metric':<20} {'Score':<10}")
    print("-" * 60)
    print(f"{'Text Classification':<30} {'Accuracy':<20} {results_text['accuracy']:<10.4f}")
    print(f"{'Headline Classification':<30} {'Accuracy':<20} {results_headline['accuracy']:<10.4f}")
    print(f"{'NLI':<30} {'Accuracy':<20} {results_nli['accuracy']:<10.4f}")
    print(f"{'Text Similarity':<30} {'Pearson Corr':<20} {results_sim['pearson']:<10.4f}")
    print(f"{'Retrieval':<30} {'MRR':<20} {results_ret['mrr']:<10.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()