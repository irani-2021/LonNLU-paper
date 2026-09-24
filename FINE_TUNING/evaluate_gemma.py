"""
Fine-tuned LLM Model Evaluation for Manipuri NLU Tasks
Evaluates Gemma 3-4B models trained with QLoRA
"""
import os
os.environ['HF_HOME'] = '/media/csenitmz/DATA/lenin/pycharm-2025.2.3/cache/'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import json
import pandas as pd
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


# ===========================
# LOAD TEST DATA
# ===========================

def read_test_data():
    """Load all test datasets"""

    # Text Classification
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/content_classification/test_content_dataset.json", 'r',
              encoding='utf-8') as f:
        data = json.load(f)

    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/content_classification/train_content_dataset.json", 'r',
              encoding='utf-8') as f:
        train_data = json.load(f)
    unique_labels = sorted(set(item['label'] for item in train_data))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    text_classification_data = [(item['text'], label2id[item['label']]) for item in data]
    print("✅ Text Classification test samples:", len(text_classification_data))

    # Headline Classification
    df_test = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/News_headlineclassification_test.csv")
    df_train = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/News_headlineclassification_train.csv")
    label2id_h = {label: idx for idx, label in enumerate(sorted(df_train['Category'].unique()))}
    df_test['label_id'] = df_test['Category'].map(label2id_h)
    headline_classification_data = list(zip(df_test['Headlines'], df_test['label_id']))
    print("✅ Headline Classification test samples:", len(headline_classification_data))

    # NLI
    df_nli_test = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/NLI/test_nli1.csv")
    df_nli_train = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/NLI/train_nli1.csv")
    label2id_n = {label: idx for idx, label in enumerate(sorted(df_nli_train['Label'].unique()))}
    df_nli_test['Label_id'] = df_nli_test['Label'].map(label2id_n)
    nli_data = list(zip(df_nli_test['Premise'], df_nli_test['Hypothesis'], df_nli_test['Label_id']))
    print("✅ NLI test samples:", len(nli_data))

    # Text Similarity
    df_sim = pd.read_csv(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/sentence_similarity/semanticsimilarity_test.csv")
    similarity_data = list(zip(df_sim['SENTENCE1'], df_sim['SENTENCE2'], df_sim['SCORE']))
    print("✅ Similarity test samples:", len(similarity_data))

    # Retrieval
    with open(r"/media/csenitmz/DATA/lenin/NLU_Mni/set_up/retrieval_test.json", 'r', encoding='utf-8') as f:
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

        query_enc = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        pos_enc = self.tokenizer(
            positive,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

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
# MODEL LOADERS
# ===========================

# --- replace load_llama_classifier, load_llama_similarity, load_llama_retrieval with these ---

def load_llama_classifier(base_model_name, lora_path, head_path, num_classes, device_ids):
    """Load model with LoRA - rely on automatic device_map inference to avoid mismatch errors"""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Ask transformers/accelerate to infer a device map automatically.
    # This avoids hardcoding module names which may differ for gemma/llama remote code.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",           # <-- let HF/accelerate infer mapping
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # wrap with PEFT
    model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")
    model.eval()

    # get hidden size
    config = model.config
    # hidden_size = getattr(config, 'hidden_dim', getattr(config, 'hidden_size', None))
    # if hidden_size is None:
    #     raise ValueError("Couldn't determine hidden size from model config.")


    #########################
    hidden_size = None
    # Prefer the canonical hidden_size if present
    if hasattr(config, "hidden_size") and isinstance(config.hidden_size, int):
        hidden_size = int(config.hidden_size)

    # Otherwise try hidden_dim
    if hidden_size is None and hasattr(config, "hidden_dim") and isinstance(config.hidden_dim, int):
        hidden_size = int(config.hidden_dim)

    # Final fallback: PROBE the model with a tiny forward pass to get last_hidden_state dim
    if hidden_size is None:
        try:
            # make a 1-token dummy batch on CPU (model has device_map="auto")
            probe_ids = torch.ones(1, 2, dtype=torch.long)  # 2 tokens to avoid -1 indexing issues
            probe_mask = torch.ones_like(probe_ids)
            with torch.no_grad():
                out = base_model.gemma(probe_ids, attention_mask=probe_mask,
                                 output_hidden_states=True, return_dict=True, use_cache=False)
            hidden_size = int(out.hidden_states[-1].size(-1))
            del out
        except Exception:
            hidden_size = 2560  # safe fallback for Gemma-3-4B

    print(f"✅ Using hidden size: {hidden_size}")

    ############################

    # put classifier on last available GPU from device_ids (fallback to cuda:0 if empty)
    last_gpu = f'cuda:{device_ids[-1]}' if device_ids else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden_size, num_classes)
    ).to(last_gpu)

    classifier.load_state_dict(torch.load(head_path, map_location=last_gpu))
    classifier = classifier.to(dtype=torch.bfloat16)
    classifier.eval()

    return model, classifier, last_gpu


def load_llama_similarity(base_model_name, lora_path, head_path, device_ids):
    """Load similarity model with LoRA using auto device mapping"""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")
    model.eval()

    config = model.config
    # hidden_size = getattr(config, 'hidden_dim', getattr(config, 'hidden_size', None))
    # if hidden_size is None:
    #     raise ValueError("Couldn't determine hidden size from model config.")

    #########################
    hidden_size = None
    # Prefer the canonical hidden_size if present
    if hasattr(config, "hidden_size") and isinstance(config.hidden_size, int):
        hidden_size = int(config.hidden_size)

    # Otherwise try hidden_dim
    if hidden_size is None and hasattr(config, "hidden_dim") and isinstance(config.hidden_dim, int):
        hidden_size = int(config.hidden_dim)

    # Final fallback: PROBE the model with a tiny forward pass to get last_hidden_state dim
    if hidden_size is None:
        try:
            # make a 1-token dummy batch on CPU (model has device_map="auto")
            probe_ids = torch.ones(1, 2, dtype=torch.long)  # 2 tokens to avoid -1 indexing issues
            probe_mask = torch.ones_like(probe_ids)
            with torch.no_grad():
                out = base_model.gemma(probe_ids, attention_mask=probe_mask,
                                       output_hidden_states=True, return_dict=True, use_cache=False)
            hidden_size = int(out.hidden_states[-1].size(-1))
            del out
        except Exception:
            hidden_size = 2560  # safe fallback for Gemma-3-4B

    print(f"✅ Using hidden size: {hidden_size}")

    ############################

    last_gpu = f'cuda:{device_ids[-1]}' if device_ids else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    regressor = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden_size, 1)
    ).to(last_gpu)

    regressor.load_state_dict(torch.load(head_path, map_location=last_gpu))
    regressor = regressor.to(dtype=torch.bfloat16)
    regressor.eval()

    return model, regressor, last_gpu


def load_llama_retrieval(base_model_name, lora_path, device_ids):
    """Load retrieval model with LoRA using auto device mapping"""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")
    model.eval()

    last_gpu = f'cuda:{device_ids[-1]}' if device_ids else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    return model, last_gpu



# ===========================
# HELPER FUNCTIONS
# ===========================

# def get_hidden_states(model, input_ids, attention_mask, device=None):
#     # Automatically detect the device of the embedding layer
#     try:
#         embed_device = model.model.embed_tokens.weight.device
#     except AttributeError:
#         # For models wrapped differently (e.g., PeftModel, Gemma)
#         embed_device = next(model.parameters()).device
#
#     input_ids = input_ids.to(embed_device)
#     attention_mask = attention_mask.to(embed_device)
#
#     with torch.no_grad():
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True
#         )
#
#     hidden_states = outputs.hidden_states[-1]
#     pooled_hidden = hidden_states.mean(dim=1)
#     return pooled_hidden
def get_hidden_states(model, input_ids, attention_mask, device=None):
    # Automatically detect the embedding layer's device
    try:
        embed_device = model.model.embed_tokens.weight.device
    except AttributeError:
        embed_device = next(model.parameters()).device

    input_ids = input_ids.to(embed_device)
    attention_mask = attention_mask.to(embed_device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

    hidden_states = outputs.hidden_states[-1]
    pooled_hidden = hidden_states.mean(dim=1)
    return pooled_hidden


# ===========================
# EVALUATION FUNCTIONS
# ===========================

def evaluate_classification(model, classifier, test_loader, last_gpu):
    model.eval()
    classifier.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # --- 1️⃣ Get hidden states from model ---
            hidden = get_hidden_states(model, input_ids, attention_mask, last_gpu)

            # --- 2️⃣ Ensure hidden and classifier are on same device ---
            classifier_device = next(classifier.parameters()).device
            hidden = hidden.to(classifier_device)

            # --- 3️⃣ Now safe to run classifier ---
            logits = classifier(hidden)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
            'confusion_matrix': cm
        }

def evaluate_similarity(model, regressor, test_loader, last_gpu):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].float()

            # ---- 1️⃣ Extract embeddings from Gemma ----
            hidden = get_hidden_states(model, input_ids, attention_mask)

            # ---- 2️⃣ Ensure hidden & regressor on SAME device ----
            reg_dev = next(regressor.parameters()).device
            hidden = hidden.to(reg_dev)
            labels = labels.to(reg_dev)

            # ---- 3️⃣ Now safe: run regressor ----
            scores = regressor(hidden).squeeze().float()  # convert BF16→FP32

            # ---- 4️⃣ Move to CPU for numpy ----
            all_preds.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    spearman_corr, _ = spearmanr(all_labels, all_preds)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }
def evaluate_retrieval(model, test_loader, last_gpu):
    """Evaluate retrieval model"""
    ranks = []
    mrr_scores = []
    recall_at_1 = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            query_ids = batch['query_input_ids']
            query_mask = batch['query_attention_mask']
            pos_ids = batch['pos_input_ids']
            pos_mask = batch['pos_attention_mask']
            neg_ids = batch['neg_input_ids']
            neg_mask = batch['neg_attention_mask']

            batch_size = query_ids.size(0)
            num_negatives = neg_ids.size(1)

            # Encode query and positive
            query_emb = get_hidden_states(model, query_ids, query_mask, last_gpu)
            pos_emb = get_hidden_states(model, pos_ids, pos_mask, last_gpu)

            for i in range(batch_size):
                q_emb = query_emb[i].unsqueeze(0)
                p_emb = pos_emb[i].unsqueeze(0)

                pos_sim = F.cosine_similarity(q_emb, p_emb).item()

                neg_sims = []
                for j in range(num_negatives):
                    n_emb = get_hidden_states(
                        model,
                        neg_ids[i, j, :].unsqueeze(0),
                        neg_mask[i, j, :].unsqueeze(0),
                        last_gpu
                    )
                    neg_sim = F.cosine_similarity(q_emb, n_emb).item()
                    neg_sims.append(neg_sim)

                all_sims = [pos_sim] + neg_sims
                sorted_indices = np.argsort(all_sims)[::-1]
                rank = np.where(sorted_indices == 0)[0][0] + 1

                ranks.append(rank)
                mrr_scores.append(1.0 / rank)

                if rank <= 1:
                    recall_at_1 += 1

    total_samples = len(ranks)

    return {
        'mrr': np.mean(mrr_scores),
        'recall@1': recall_at_1 / total_samples,
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks)
    }


# ===========================
# MAIN EVALUATION
# ===========================

def main():
    # Configuration
    MODEL_TYPE = "gemma"
    # MODEL_TYPE = "gemma"  # or "gemma" NOt WORKING for GEMMA

    if MODEL_TYPE == "llama":
        print("=" * 60)
        print("Evaluating Llama 3.1-8B Fine-tuned Models")
        print("=" * 60)
        BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
        MODEL_PREFIX = "llama"
    else:
        print("=" * 60)
        print("Evaluating Gemma 3-4B Fine-tuned Models")
        print("=" * 60)
        BASE_MODEL_NAME = "google/gemma-3-4b-pt"
        MODEL_PREFIX = "gemma"

    BATCH_SIZE = 16
    MAX_LENGTH = 128

    if not torch.cuda.is_available():
        print("❌ No GPU available!")
        return

    device_ids = [0, 1, 2, 3] if torch.cuda.device_count() >= 4 else list(range(torch.cuda.device_count()))
    print(f"Using GPUs: {device_ids}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load test data
    print("\nLoading test data...")
    test_datasets = read_test_data()

    results_summary = {}

    # ===========================
    # Task 1: Text Classification
    # ===========================
    print("\n" + "=" * 60)
    print("TASK 1: Text Classification Evaluation")
    print("=" * 60)

    texts = [item[0] for item in test_datasets['text_classification']]
    labels = [item[1] for item in test_datasets['text_classification']]

    test_dataset = ClassificationDataset(texts, labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, classifier, last_gpu = load_llama_classifier(
        BASE_MODEL_NAME,
        f'./{MODEL_PREFIX}_text_classification_lora',
        f'./{MODEL_PREFIX}_text_classification_head.pt',
        num_classes=3,
        device_ids=device_ids
    )

    results_text = evaluate_classification(model, classifier, test_loader, last_gpu)

    print(f"\nResults:")
    print(f"  Accuracy:  {results_text['accuracy']:.4f}")
    print(f"  Precision: {results_text['precision']:.4f}")
    print(f"  Recall:    {results_text['recall']:.4f}")
    print(f"  F1 Score:  {results_text['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results_text['confusion_matrix'])

    results_summary['Text Classification'] = results_text['accuracy']

    del model, classifier
    torch.cuda.empty_cache()

    # ===========================
    # Task 2: Headline Classification
    # ===========================
    print("\n" + "=" * 60)
    print("TASK 2: Headline Classification Evaluation")
    print("=" * 60)

    headlines = [item[0] for item in test_datasets['headline_classification']]
    h_labels = [item[1] for item in test_datasets['headline_classification']]

    test_hl_dataset = ClassificationDataset(headlines, h_labels, tokenizer, MAX_LENGTH)
    test_hl_loader = DataLoader(test_hl_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, classifier, last_gpu = load_llama_classifier(
        BASE_MODEL_NAME,
        f'./{MODEL_PREFIX}_headline_classification_lora',
        f'./{MODEL_PREFIX}_headline_classification_head.pt',
        num_classes=4,
        device_ids=device_ids
    )

    results_headline = evaluate_classification(model, classifier, test_hl_loader, last_gpu)

    print(f"\nResults:")
    print(f"  Accuracy:  {results_headline['accuracy']:.4f}")
    print(f"  Precision: {results_headline['precision']:.4f}")
    print(f"  Recall:    {results_headline['recall']:.4f}")
    print(f"  F1 Score:  {results_headline['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results_headline['confusion_matrix'])

    results_summary['Headline Classification'] = results_headline['accuracy']

    del model, classifier
    torch.cuda.empty_cache()

    # ===========================
    # Task 3: NLI
    # ===========================
    print("\n" + "=" * 60)
    print("TASK 3: Natural Language Inference Evaluation")
    print("=" * 60)

    premises = [item[0] for item in test_datasets['nli']]
    hypotheses = [item[1] for item in test_datasets['nli']]
    nli_labels = [item[2] for item in test_datasets['nli']]

    test_nli_dataset = NLIDataset(premises, hypotheses, nli_labels, tokenizer, MAX_LENGTH)
    test_nli_loader = DataLoader(test_nli_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, classifier, last_gpu = load_llama_classifier(
        BASE_MODEL_NAME,
        f'./{MODEL_PREFIX}_nli_lora',
        f'./{MODEL_PREFIX}_nli_head.pt',
        num_classes=3,
        device_ids=device_ids
    )

    results_nli = evaluate_classification(model, classifier, test_nli_loader, last_gpu)

    print(f"\nResults:")
    print(f"  Accuracy:  {results_nli['accuracy']:.4f}")
    print(f"  Precision: {results_nli['precision']:.4f}")
    print(f"  Recall:    {results_nli['recall']:.4f}")
    print(f"  F1 Score:  {results_nli['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(results_nli['confusion_matrix'])

    results_summary['NLI'] = results_nli['accuracy']

    del model, classifier
    torch.cuda.empty_cache()

    # ===========================
    # Task 4: Text Similarity
    # ===========================
    print("\n" + "=" * 60)
    print("TASK 4: Text Similarity Evaluation")
    print("=" * 60)

    text1_list = [item[0] for item in test_datasets['similarity']]
    text2_list = [item[1] for item in test_datasets['similarity']]
    sim_scores = [item[2] for item in test_datasets['similarity']]

    test_sim_dataset = SimilarityDataset(text1_list, text2_list, sim_scores, tokenizer, MAX_LENGTH)
    test_sim_loader = DataLoader(test_sim_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, regressor, last_gpu = load_llama_similarity(
        BASE_MODEL_NAME,
        f'./{MODEL_PREFIX}_similarity_lora',
        f'./{MODEL_PREFIX}_similarity_head.pt',
        device_ids=device_ids
    )

    results_sim = evaluate_similarity(model, regressor, test_sim_loader, last_gpu)

    print(f"\nResults:")
    print(f"  MSE:              {results_sim['mse']:.4f}")
    print(f"  MAE:              {results_sim['mae']:.4f}")
    print(f"  RMSE:             {results_sim['rmse']:.4f}")
    print(f"  Pearson Corr:     {results_sim['pearson']:.4f}")
    print(f"  Spearman Corr:    {results_sim['spearman']:.4f}")

    results_summary['Text Similarity'] = results_sim['pearson']

    del model, regressor
    torch.cuda.empty_cache()

    # ===========================
    # Task 5: Retrieval
    # ===========================
    print("\n" + "=" * 60)
    print("TASK 5: Retrieval Evaluation")
    print("=" * 60)

    queries = [item[0] for item in test_datasets['retrieval']]
    positives = [item[1] for item in test_datasets['retrieval']]
    negatives = [item[2] for item in test_datasets['retrieval']]

    test_ret_dataset = RetrievalDataset(queries, positives, negatives, tokenizer, MAX_LENGTH)
    test_ret_loader = DataLoader(test_ret_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, last_gpu = load_llama_retrieval(
        BASE_MODEL_NAME,
        f'./{MODEL_PREFIX}_retrieval_lora',
        device_ids=device_ids
    )

    results_ret = evaluate_retrieval(model, test_ret_loader, last_gpu)

    print(f"\nResults:")
    print(f"  MRR (Mean Reciprocal Rank): {results_ret['mrr']:.4f}")
    print(f"Recall@1:                   {results_ret['recall@1']:.4f}")
    print(f"  Mean Rank:                  {results_ret['mean_rank']:.2f}")
    print(f"  Median Rank:                {results_ret['median_rank']:.2f}")

    results_summary['Retrieval'] = results_ret['mrr']

    del model
    torch.cuda.empty_cache()

    # ===========================
    # Summary
    # ===========================
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<30} {'Primary Metric':<20} {'Score':<10}")
    print("-" * 60)
    print(f"{'Text Classification':<30} {'Accuracy':<20} {results_summary['Text Classification']:<10.4f}")
    print(f"{'Headline Classification':<30} {'Accuracy':<20} {results_summary['Headline Classification']:<10.4f}")
    print(f"{'NLI':<30} {'Accuracy':<20} {results_summary['NLI']:<10.4f}")
    print(f"{'Text Similarity':<30} {'Pearson Corr':<20} {results_summary['Text Similarity']:<10.4f}")
    print(f"{'Retrieval':<30} {'MRR':<20} {results_summary['Retrieval']:<10.4f}")
    print("=" * 60)

    # Calculate average score (normalized)
    avg_score = (
        results_summary['Text Classification'] +
        results_summary['Headline Classification'] +
        results_summary['NLI'] +
        results_summary['Text Similarity'] +
        results_summary['Retrieval']
    ) / 5

    print(f"\n{'Average Score (all tasks):':<30} {avg_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()