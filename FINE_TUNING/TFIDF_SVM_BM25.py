import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 📊 Retrieval Metrics
# ============================================================

def mean_reciprocal_rank(all_scores):
    rr = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        rank_pos = np.where(ranks == 0)[0][0] + 1
        rr.append(1.0 / rank_pos)
    return np.mean(rr)

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
    return np.mean(precisions)

def mean_average_precision(all_scores):
    avg_precs = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        rank_pos = np.where(ranks == 0)[0][0] + 1
        avg_precs.append(1.0 / rank_pos)
    return np.mean(avg_precs)

# ============================================================
# 🧮 Text Similarity Metrics
# ============================================================

def evaluate_similarity(y_true, y_pred):
    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Pearson: {pearson:.4f} | Spearman: {spearman:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")

# ============================================================
# 🔍 BM25 + PRF Retrieval
# ============================================================

def bm25_prf_retrieval(retrieval_test, prf_k=2):
    all_scores = []
    for q, p, negs in retrieval_test:
        # Tokenize
        docs = [p] + negs
        tokenized_corpus = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = q.split()

        # Initial BM25 scores
        init_scores = bm25.get_scores(tokenized_query)
        topk_idx = np.argsort(init_scores)[::-1][:prf_k]
        pseudo_rel = [tokenized_corpus[i] for i in topk_idx]

        # Pseudo relevance feedback
        expanded_query = tokenized_query[:]
        for doc in pseudo_rel:
            expanded_query.extend(doc)
        final_scores = bm25.get_scores(expanded_query)

        all_scores.append([final_scores[0]] + final_scores[1:].tolist())

    mrr = mean_reciprocal_rank(all_scores)
    rec1 = recall_at_k(all_scores, k=1)
    prec1 = precision_at_k(all_scores, k=1)
    map_score = mean_average_precision(all_scores)

    print(f"MRR: {mrr:.4f} | Recall@1: {rec1:.4f} | Precision@1: {prec1:.4f} | MAP: {map_score:.4f}")

# ============================================================
# ⚙️ Main Baseline Runner
# ============================================================

def run_baseline(data):
    print("\n================= TEXT CLASSIFICATION (SVM) =================")
    tc_train = data['train']['text_classification']
    tc_test = data['test']['text_classification']

    X_train, y_train = zip(*tc_train)
    X_test, y_test = zip(*tc_test)

    vectorizer = TfidfVectorizer(max_features=6000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds))

    # -----------------------------------------------------
    print("\n================= HEADLINE CLASSIFICATION (SVM) =================")
    hc_train = data['train']['headline_classification']
    hc_test = data['test']['headline_classification']

    X_train, y_train = zip(*hc_train)
    X_test, y_test = zip(*hc_test)

    vectorizer = TfidfVectorizer(max_features=6000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds))

    # -----------------------------------------------------
    print("\n================= NLI (SVM) =================")
    nli_train = data['train']['nli']
    nli_test = data['test']['nli']

    X_train = [a + " " + b for a, b, _ in nli_train]
    y_train = [label for _, _, label in nli_train]
    X_test = [a + " " + b for a, b, _ in nli_test]
    y_test = [label for _, _, label in nli_test]

    vectorizer = TfidfVectorizer(max_features=8000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds))

    # -----------------------------------------------------
    print("\n================= TEXT SIMILARITY (SVR) =================")
    sim_train = data['train']['similarity']
    sim_test = data['test']['similarity']

    X_train = [a + " " + b for a, b, _ in sim_train]
    y_train = [score for _, _, score in sim_train]
    X_test = [a + " " + b for a, b, _ in sim_test]
    y_test = [score for _, _, score in sim_test]

    vectorizer = TfidfVectorizer(max_features=8000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = SVR(kernel='linear')
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    evaluate_similarity(y_test, preds)

    # -----------------------------------------------------
    print("\n================= RETRIEVAL (BM25 + PRF) =================")
    bm25_prf_retrieval(data['test']['retrieval'], prf_k=2)

# ============================================================
# 🧾 Example Data Loader (same as before)
# ============================================================

def read_data():
    """Loads both train and test data (modify paths)."""
    with open(r"E:\NLU\content_classification\train_content_dataset.json", 'r', encoding='utf-8') as f:
        train_content = json.load(f)
    with open(r"E:\NLU\content_classification\test_content_dataset.json", 'r', encoding='utf-8') as f:
        test_content = json.load(f)
    text_classification_train = [(x['text'], x['label']) for x in train_content]
    text_classification_test = [(x['text'], x['label']) for x in test_content]

    df_train_h = pd.read_csv(r"E:\NLU\News_headlineclassification_train.csv")
    df_test_h = pd.read_csv(r"E:\NLU\News_headlineclassification_test.csv")
    lbl2id = {l: i for i, l in enumerate(sorted(df_train_h['Category'].unique()))}
    df_train_h['label_id'] = df_train_h['Category'].map(lbl2id)
    df_test_h['label_id'] = df_test_h['Category'].map(lbl2id)
    headline_classification_train = list(zip(df_train_h['Headlines'], df_train_h['label_id']))
    headline_classification_test = list(zip(df_test_h['Headlines'], df_test_h['label_id']))

    df_train_nli = pd.read_csv(r"E:\NLU\NLI\train_nli1.csv")
    df_test_nli = pd.read_csv(r"E:\NLU\NLI\test_nli1.csv")
    lbl2id = {l: i for i, l in enumerate(sorted(df_train_nli['Label'].unique()))}
    df_train_nli['Label_id'] = df_train_nli['Label'].map(lbl2id)
    df_test_nli['Label_id'] = df_test_nli['Label'].map(lbl2id)
    nli_train = list(zip(df_train_nli['Premise'], df_train_nli['Hypothesis'], df_train_nli['Label_id']))
    nli_test = list(zip(df_test_nli['Premise'], df_test_nli['Hypothesis'], df_test_nli['Label_id']))

    df_train_sim = pd.read_csv(r"E:\NLU\sentence_similarity\semanticsimilarity_train.csv")
    df_test_sim = pd.read_csv(r"E:\NLU\sentence_similarity\semanticsimilarity_test.csv")
    sim_train = list(zip(df_train_sim['SENTENCE1'], df_train_sim['SENTENCE2'], df_train_sim['SCORE']))
    sim_test = list(zip(df_test_sim['SENTENCE1'], df_test_sim['SENTENCE2'], df_test_sim['SCORE']))

    with open(r"E:\NLU\retrieval_train.json", 'r', encoding='utf-8') as f:
        train_ret = json.load(f)
    with open(r"E:\NLU\retrieval_test.json", 'r', encoding='utf-8') as f:
        test_ret = json.load(f)
    ret_train = [(i['query'], i['positive'], [i['negatives']['random'], i['negatives']['tfidf'], i['negatives']['bm25']]) for i in train_ret]
    ret_test = [(i['query'], i['positive'], [i['negatives']['random'], i['negatives']['tfidf'], i['negatives']['bm25']]) for i in test_ret]

    return {
        'train': {
            'text_classification': text_classification_train,
            'headline_classification': headline_classification_train,
            'nli': nli_train,
            'similarity': sim_train,
            'retrieval': ret_train
        },
        'test': {
            'text_classification': text_classification_test,
            'headline_classification': headline_classification_test,
            'nli': nli_test,
            'similarity': sim_test,
            'retrieval': ret_test
        }
    }

# ============================================================
if __name__ == "__main__":
    data = read_data()
    run_baseline(data)
