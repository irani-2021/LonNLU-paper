"""
BiLSTM + Attention baseline for Manipuri NLU tasks - FIXED VERSION

Fixed: Label encoding for text classification tasks
"""

import os
import json
import math
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


#################### Added #################

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

#########################################

# ---------------------------
# Utility: metrics for retrieval
# ---------------------------
def mean_reciprocal_rank(all_scores):
    rr = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        rank_pos = np.where(ranks == 0)[0][0] + 1
        rr.append(1.0 / rank_pos)
    return float(np.mean(rr))

def recall_at_k(all_scores, k=3):
    hits = 0
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        if 0 in ranks[:k]:
            hits += 1
    return hits / len(all_scores)

def precision_at_k(all_scores, k=3):
    precisions = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        top_k = ranks[:k]
        precisions.append(1 if 0 in top_k else 0)
    return float(np.mean(precisions))

def mean_average_precision(all_scores):
    avg_precs = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        rank_pos = np.where(ranks == 0)[0][0] + 1
        avg_precs.append(1.0 / rank_pos)
    return float(np.mean(avg_precs))


# ---------------------------
# Tokenizer & Vocab
# ---------------------------
def whitespace_tokenize(text: str):
    return text.strip().split()

class Vocab:
    def __init__(self, min_freq=1, unk_token="<UNK>", pad_token="<PAD>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.freqs = Counter()
        self.itos = []
        self.stoi = {}

    def update_from_texts(self, texts: List[str]):
        for t in texts:
            toks = whitespace_tokenize(t)
            self.freqs.update(toks)

    def build_vocab(self):
        tokens = [tok for tok, freq in self.freqs.items() if freq >= self.min_freq]
        tokens = sorted(tokens)
        self.itos = [self.pad_token, self.unk_token] + tokens
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def encode(self, tokens: List[str]):
        return [self.stoi.get(t, self.stoi[self.unk_token]) for t in tokens]

    def __len__(self):
        return len(self.itos)


# ---------------------------
# PyTorch Datasets
# ---------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, data: List[Tuple[str,int]], vocab: Vocab):
        self.samples = data
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        toks = whitespace_tokenize(text)
        ids = torch.LongTensor(self.vocab.encode(toks))
        # Ensure label is an integer
        label_int = int(label) if not isinstance(label, int) else label
        return ids, label_int

class NLIDataset(Dataset):
    def __init__(self, data: List[Tuple[str,str,int]], vocab: Vocab):
        self.samples = data
        self.vocab = vocab

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, h, label = self.samples[idx]
        p_ids = torch.LongTensor(self.vocab.encode(whitespace_tokenize(p)))
        h_ids = torch.LongTensor(self.vocab.encode(whitespace_tokenize(h)))
        label_int = int(label) if not isinstance(label, int) else label
        return p_ids, h_ids, label_int

class SimilarityDataset(Dataset):
    def __init__(self, data: List[Tuple[str,str,float]], vocab: Vocab):
        self.samples = data
        self.vocab = vocab
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s1, s2, score = self.samples[idx]
        s1_ids = torch.LongTensor(self.vocab.encode(whitespace_tokenize(s1)))
        s2_ids = torch.LongTensor(self.vocab.encode(whitespace_tokenize(s2)))
        return s1_ids, s2_ids, float(score)

class RetrievalDataset(Dataset):
    def __init__(self, data: List[Tuple[str,str,List[str]]], vocab: Vocab):
        self.samples = data
        self.vocab = vocab
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        q, pos, negs = self.samples[idx]
        q_ids = torch.LongTensor(self.vocab.encode(whitespace_tokenize(q)))
        pos_ids = torch.LongTensor(self.vocab.encode(whitespace_tokenize(pos)))
        negs_ids = [torch.LongTensor(self.vocab.encode(whitespace_tokenize(n))) for n in negs]
        return q_ids, pos_ids, negs_ids


# collate helpers
def collate_pad(batch):
    seqs, labels = zip(*batch)
    lengths = torch.LongTensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return padded, lengths, labels

def collate_nli(batch):
    ps, hs, labels = zip(*batch)
    p_lens = torch.LongTensor([len(x) for x in ps])
    h_lens = torch.LongTensor([len(x) for x in hs])
    ps_padded = pad_sequence(ps, batch_first=True, padding_value=0)
    hs_padded = pad_sequence(hs, batch_first=True, padding_value=0)
    return ps_padded, p_lens, hs_padded, h_lens, torch.LongTensor(labels)

def collate_sim(batch):
    s1s, s2s, scores = zip(*batch)
    l1 = torch.LongTensor([len(x) for x in s1s])
    l2 = torch.LongTensor([len(x) for x in s2s])
    s1p = pad_sequence(s1s, batch_first=True, padding_value=0)
    s2p = pad_sequence(s2s, batch_first=True, padding_value=0)
    return s1p, l1, s2p, l2, torch.tensor(scores, dtype=torch.float32)

def collate_retrieval(batch):
    qs, pos, neglists = zip(*batch)
    q_lens = torch.LongTensor([len(x) for x in qs])
    pos_lens = torch.LongTensor([len(x) for x in pos])
    qs_p = pad_sequence(qs, batch_first=True, padding_value=0)
    pos_p = pad_sequence(pos, batch_first=True, padding_value=0)
    n_negs = len(neglists[0])
    negs_p = []
    negs_lens = []
    for i in range(n_negs):
        ith = [nl[i] for nl in neglists]
        lengths = torch.LongTensor([len(x) for x in ith])
        padded = pad_sequence(ith, batch_first=True, padding_value=0)
        negs_p.append(padded)
        negs_lens.append(lengths)
    return qs_p, q_lens, pos_p, pos_lens, negs_p, negs_lens


# ---------------------------
# Model: BiLSTM + Attention encoder
# ---------------------------
class BiLSTMAttentionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_size=256, n_layers=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers>1 else 0.0
        )
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.attn_proj = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        attn_logits = self.attn_proj(out).squeeze(-1)
        max_len = out.size(1)
        mask = (torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1))
        attn_logits[~mask] = float('-inf')
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        weighted = (out * attn_weights).sum(dim=1)
        return self.dropout(weighted), attn_weights


# ---------------------------
# Heads: classification & regression
# ---------------------------
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, n_classes)
        )
    def forward(self, x): return self.fc(x)

class SimilarityRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim//2, 1)
        )
    def forward(self, x): return self.fc(x).squeeze(-1)


# ---------------------------
# Training helpers
# ---------------------------
def train_epoch_classification(encoder, head, loader, opt_enc, opt_head, device):
    encoder.train(); head.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        emb, _ = encoder(x, lengths)
        logits = head(emb)
        loss = criterion(logits, y)
        opt_enc.zero_grad(); opt_head.zero_grad()
        loss.backward()
        opt_enc.step(); opt_head.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_classification(encoder, head, loader, device):
    encoder.eval(); head.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths = x.to(device), lengths.to(device)
            emb, _ = encoder(x, lengths)
            logits = head(emb)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            ys.extend(y.numpy().tolist())
    acc = accuracy_score(ys, preds)
    p, r, f1, _ = precision_recall_fscore_support(ys, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(ys, preds)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'cm': cm, 'y_true': ys, 'y_pred': preds}

def train_epoch_svr(encoder, head, loader, opt_enc, opt_head, device):
    encoder.train(); head.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for s1, l1, s2, l2, scores in loader:
        s1, s2 = s1.to(device), s2.to(device)
        l1, l2 = l1.to(device), l2.to(device)
        scores = scores.to(device)
        e1, _ = encoder(s1, l1)
        e2, _ = encoder(s2, l2)
        comb = torch.cat([e1, e2, torch.abs(e1-e2), e1*e2], dim=1)
        pred = head(comb)
        loss = criterion(pred, scores)
        opt_enc.zero_grad(); opt_head.zero_grad()
        loss.backward()
        opt_enc.step(); opt_head.step()
        total_loss += loss.item() * s1.size(0)
    return total_loss / len(loader.dataset)

def eval_similarity(encoder, head, loader, device):
    encoder.eval(); head.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for s1, l1, s2, l2, scores in loader:
            s1, s2 = s1.to(device), s2.to(device)
            l1, l2 = l1.to(device), l2.to(device)
            e1, _ = encoder(s1, l1)
            e2, _ = encoder(s2, l2)
            comb = torch.cat([e1, e2, torch.abs(e1-e2), e1*e2], dim=1)
            pred = head(comb).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(scores.numpy().tolist())
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
    spearman_r = spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else 0.0
    return {'mse': mse, 'mae': mae, 'pearson': pearson_r, 'spearman': spearman_r, 'y_true': y_true, 'y_pred': y_pred}


# ---------------------------
# Retrieval evaluation using embedding similarity
# ---------------------------
def evaluate_retrieval_embeddings(encoder, device, retrieval_test, vocab, batch_size=64):
    encoder.eval()
    from torch.utils.data import DataLoader

    queries = [q for q, _, _ in retrieval_test]
    pos = [p for _, p, _ in retrieval_test]
    negs = [negs for _, _, negs in retrieval_test]

    def make_loader(sentences):
        seqs = [torch.LongTensor(vocab.encode(whitespace_tokenize(s))) for s in sentences]
        lengths = [len(s) for s in seqs]
        dataset = list(zip(seqs, lengths))
        def coll(batch):
            seqs_b, lens_b = zip(*batch)
            padded = pad_sequence(seqs_b, batch_first=True, padding_value=0)
            return padded, torch.LongTensor(lens_b)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=coll)

    q_loader = make_loader(queries)
    pos_loader = make_loader(pos)
    n_negs = len(negs[0])
    negs_flat = []
    for i in range(n_negs):
        negs_flat.append([n[i] for n in negs])
    neg_loaders = [make_loader(lst) for lst in negs_flat]

    def encode_loader(dl):
        emb_list = []
        with torch.no_grad():
            for xb, lengths in dl:
                xb, lengths = xb.to(device), lengths.to(device)
                e, _ = encoder(xb, lengths)
                emb_list.append(e.cpu())
        return torch.cat(emb_list, dim=0).numpy()

    q_emb = encode_loader(q_loader)
    pos_emb = encode_loader(pos_loader)
    neg_embs = [encode_loader(nl) for nl in neg_loaders]

    all_scores = []
    for i in range(len(queries)):
        cand_embs = [pos_emb[i]] + [ne[i] for ne in neg_embs]
        qv = q_emb[i]
        sims = [ (qv @ c) / (np.linalg.norm(qv) * (np.linalg.norm(c)+1e-8) + 1e-8) for c in cand_embs ]
        all_scores.append(np.array(sims))

    mrr = mean_reciprocal_rank(all_scores)
    rec1 = recall_at_k(all_scores, k=1)
    prec1 = precision_at_k(all_scores, k=1)
    map_sc = mean_average_precision(all_scores)
    return {'mrr': mrr, 'recall@1': rec1, 'precision@1': prec1, 'map': map_sc}


# ---------------------------
# Entire pipeline runner
# ---------------------------
def run_all_bilstm(read_data_fn,
                   emb_dim=200, hidden_size=256, batch_size=32,
                   epochs_cls=6, epochs_simi=8, lr=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    data = read_data_fn()
    train = data['train']; test = data['test']

    # Build vocab from all training text sources
    vocab = Vocab(min_freq=1)
    texts = []
    texts += [t for t, _ in train['text_classification']]
    texts += [t for t, _ in train['headline_classification']]
    texts += [p for p, h, _ in train['nli']] + [h for p, h, _ in train['nli']]
    texts += [s1 for s1, s2, _ in train['similarity']] + [s2 for s1, s2, _ in train['similarity']]
    texts += [q for q, _, _ in train['retrieval']]
    texts += [p for _, p, _ in train['retrieval']]
    for _, _, negs in train['retrieval']:
        texts += list(negs)
    vocab.update_from_texts(texts)
    vocab.build_vocab()
    print("Vocab size:", len(vocab))

    # ---------------- Text classification ----------------
    print("\n=== Text Classification (BiLSTM+attn) ===")
    train_tc = train['text_classification']; test_tc = test['text_classification']
    ds_tr = TextClassificationDataset(train_tc, vocab)
    ds_te = TextClassificationDataset(test_tc, vocab)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    encoder = BiLSTMAttentionEncoder(len(vocab), emb_dim=emb_dim, hidden_size=hidden_size).to(device)
    head = ClassifierHead(hidden_size * 2, n_classes=len(set([l for _, l in train_tc]))).to(device)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=lr)
    opt_head = torch.optim.Adam(head.parameters(), lr=lr)

    for ep in range(epochs_cls):
        loss = train_epoch_classification(encoder, head, dl_tr, opt_enc, opt_head, device)
        print(f"Epoch {ep+1}/{epochs_cls} train_loss: {loss:.4f}")
    cls_res = eval_classification(encoder, head, dl_te, device)
    print("Text classification - Accuracy:", cls_res['accuracy'], "F1:", cls_res['f1'])
    print("Confusion matrix:\n", cls_res['cm'])

    # ---------------- Headline classification ----------------
    print("\n=== Headline Classification (BiLSTM+attn) ===")
    train_hc = train['headline_classification']; test_hc = test['headline_classification']
    ds_tr = TextClassificationDataset(train_hc, vocab)
    ds_te = TextClassificationDataset(test_hc, vocab)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    encoder_h = BiLSTMAttentionEncoder(len(vocab), emb_dim=emb_dim, hidden_size=hidden_size).to(device)
    head_h = ClassifierHead(hidden_size * 2, n_classes=len(set([l for _, l in train_hc]))).to(device)
    opt_enc = torch.optim.Adam(encoder_h.parameters(), lr=lr)
    opt_head = torch.optim.Adam(head_h.parameters(), lr=lr)

    for ep in range(epochs_cls):
        loss = train_epoch_classification(encoder_h, head_h, dl_tr, opt_enc, opt_head, device)
        print(f"Epoch {ep+1}/{epochs_cls} train_loss: {loss:.4f}")
    hc_res = eval_classification(encoder_h, head_h, dl_te, device)
    print("Headline classification - Accuracy:", hc_res['accuracy'], "F1:", hc_res['f1'])

    # ---------------- NLI ----------------
    print("\n=== NLI (BiLSTM+attn) ===")
    train_nli = train['nli']; test_nli = test['nli']
    ds_tr = NLIDataset(train_nli, vocab); ds_te = NLIDataset(test_nli, vocab)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_nli)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_nli)

    encoder_n = BiLSTMAttentionEncoder(len(vocab), emb_dim=emb_dim, hidden_size=hidden_size).to(device)
    cls_in_dim = (hidden_size * 2) * 4
    head_n = ClassifierHead(cls_in_dim, n_classes=len(set([lbl for _, _, lbl in train_nli]))).to(device)
    opt_enc = torch.optim.Adam(encoder_n.parameters(), lr=lr)
    opt_head = torch.optim.Adam(head_n.parameters(), lr=lr)

    def train_epoch_nli(enc, head, loader, opt_enc, opt_head, device):
        enc.train(); head.train()
        totloss = 0.0
        crit = nn.CrossEntropyLoss()
        for p, plens, h, hlens, labels in loader:
            p, plens, h, hlens, labels = p.to(device), plens.to(device), h.to(device), hlens.to(device), labels.to(device)
            e_p, _ = enc(p, plens)
            e_h, _ = enc(h, hlens)
            comb = torch.cat([e_p, e_h, torch.abs(e_p-e_h), e_p*e_h], dim=1)
            logits = head(comb)
            loss = crit(logits, labels)
            opt_enc.zero_grad(); opt_head.zero_grad()
            loss.backward(); opt_enc.step(); opt_head.step()
            totloss += loss.item() * p.size(0)
        return totloss / len(loader.dataset)

    def eval_nli(enc, head, loader, device):
        enc.eval(); head.eval()
        ys, preds = [], []
        with torch.no_grad():
            for p, plens, h, hlens, labels in loader:
                p, plens, h, hlens = p.to(device), plens.to(device), h.to(device), hlens.to(device)
                e_p, _ = enc(p, plens)
                e_h, _ = enc(h, hlens)
                comb = torch.cat([e_p, e_h, torch.abs(e_p-e_h), e_p*e_h], dim=1)
                logits = head(comb)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist()); ys.extend(labels.numpy().tolist())
        acc = accuracy_score(ys, preds)
        p, r, f1, _ = precision_recall_fscore_support(ys, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(ys, preds)
        return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'cm': cm}

    for ep in range(epochs_cls):
        loss = train_epoch_nli(encoder_n, head_n, dl_tr, opt_enc, opt_head, device)
        print(f"NLI Epoch {ep+1}/{epochs_cls} train_loss: {loss:.4f}")
    nli_res = eval_nli(encoder_n, head_n, dl_te, device)
    print("NLI - Accuracy:", nli_res['accuracy'], "F1:", nli_res['f1'])

    # ---------------- Similarity ----------------
    print("\n=== Text Similarity (BiLSTM embeddings + regressor) ===")
    train_sim = train['similarity']; test_sim = test['similarity']
    ds_tr = SimilarityDataset(train_sim, vocab); ds_te = SimilarityDataset(test_sim, vocab)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_sim)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_sim)

    encoder_s = BiLSTMAttentionEncoder(len(vocab), emb_dim=emb_dim, hidden_size=hidden_size).to(device)
    sim_in_dim = (hidden_size * 2) * 4
    regressor = SimilarityRegressor(sim_in_dim).to(device)
    opt_enc = torch.optim.Adam(encoder_s.parameters(), lr=lr)
    opt_reg = torch.optim.Adam(regressor.parameters(), lr=lr)

    for ep in range(epochs_simi):
        loss = train_epoch_svr(encoder_s, regressor, dl_tr, opt_enc, opt_reg, device)
        print(f"Similarity Epoch {ep+1}/{epochs_simi} train_loss: {loss:.4f}")
    sim_res = eval_similarity(encoder_s, regressor, dl_te, device)
    print("Similarity - MSE:", sim_res['mse'], "Pearson:", sim_res['pearson'], "Spearman:", sim_res['spearman'])

    # ---------------- Retrieval ----------------
    print("\n=== Retrieval (embedding cosine over BiLSTM embeddings) ===")
    retr_test = test['retrieval']
    retr_res = evaluate_retrieval_embeddings(encoder, device, retr_test, vocab, batch_size=batch_size)
    print("Retrieval results:", retr_res)

    return {
        'text_classification': cls_res,
        'headline_classification': hc_res,
        'nli': nli_res,
        'similarity': sim_res,
        'retrieval': retr_res
    }


# ---------------------------
# Example read_data() - FIXED VERSION
# ---------------------------
def read_data():
    """
    Fixed version that ensures all labels are properly encoded as integers
    """
    # Text classification
    with open(r"E:\NLU\content_classification\train_content_dataset.json", 'r', encoding='utf-8') as f:
        train_tc_json = json.load(f)
    with open(r"E:\NLU\content_classification\test_content_dataset.json", 'r', encoding='utf-8') as f:
        test_tc_json = json.load(f)

    # Encode text classification labels
    tc_labels = sorted(set([item['label'] for item in train_tc_json]))
    tc_label2id = {lbl: i for i, lbl in enumerate(tc_labels)}
    train_tc = [(item['text'], tc_label2id[item['label']]) for item in train_tc_json]
    test_tc = [(item['text'], tc_label2id[item['label']]) for item in test_tc_json]

    # Headline classification
    df_tr_h = pd.read_csv(r"E:\NLU\News_headlineclassification_train.csv")
    df_te_h = pd.read_csv(r"E:\NLU\News_headlineclassification_test.csv")
    lbl2id_h = {lbl: i for i, lbl in enumerate(sorted(df_tr_h['Category'].unique()))}
    df_tr_h['label_id'] = df_tr_h['Category'].map(lbl2id_h)
    df_te_h['label_id'] = df_te_h['Category'].map(lbl2id_h)
    train_hc = list(zip(df_tr_h['Headlines'], df_tr_h['label_id']))
    test_hc = list(zip(df_te_h['Headlines'], df_te_h['label_id']))

    # NLI
    df_tr_n = pd.read_csv(r"E:\NLU\NLI\train_nli1.csv")
    df_te_n = pd.read_csv(r"E:\NLU\NLI\test_nli1.csv")
    lbl2id_n = {lbl: i for i, lbl in enumerate(sorted(df_tr_n['Label'].unique()))}
    df_tr_n['Label_id'] = df_tr_n['Label'].map(lbl2id_n)
    df_te_n['Label_id'] = df_te_n['Label'].map(lbl2id_n)
    train_nli = list(zip(df_tr_n['Premise'], df_tr_n['Hypothesis'], df_tr_n['Label_id']))
    test_nli = list(zip(df_te_n['Premise'], df_te_n['Hypothesis'], df_te_n['Label_id']))

    # Similarity
    df_tr_s = pd.read_csv(r"E:\NLU\sentence_similarity\semanticsimilarity_train.csv")
    df_te_s = pd.read_csv(r"E:\NLU\sentence_similarity\semanticsimilarity_test.csv")
    train_sim = list(zip(df_tr_s['SENTENCE1'], df_tr_s['SENTENCE2'], df_tr_s['SCORE']))
    test_sim = list(zip(df_te_s['SENTENCE1'], df_te_s['SENTENCE2'], df_te_s['SCORE']))

    # Retrieval
    with open(r"E:\NLU\retrieval_train.json", 'r', encoding='utf-8') as f:
        train_ret = json.load(f)
    with open(r"E:\NLU\retrieval_test.json", 'r', encoding='utf-8') as f:
        test_ret = json.load(f)
    train_retr = [(it['query'], it['positive'], [it['negatives']['random'], it['negatives']['tfidf'], it['negatives']['bm25']]) for it in train_ret]
    test_retr = [(it['query'], it['positive'], [it['negatives']['random'], it['negatives']['tfidf'], it['negatives']['bm25']]) for it in test_ret]

    return {
        'train': {
            'text_classification': train_tc,
            'headline_classification': train_hc,
            'nli': train_nli,
            'similarity': train_sim,
            'retrieval': train_retr
        },
        'test': {
            'text_classification': test_tc,
            'headline_classification': test_hc,
            'nli': test_nli,
            'similarity': test_sim,
            'retrieval': test_retr
        }
    }


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    results = run_all_bilstm(read_data, emb_dim=200, hidden_size=256, batch_size=32,
                             epochs_cls=4, epochs_simi=6, lr=1e-3)
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper()}:")
        for metric, value in task_results.items():
            if metric not in ['cm', 'y_true', 'y_pred']:
                print(f"  {metric}: {value}")