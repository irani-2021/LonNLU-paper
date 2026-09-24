import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============ Retrieval Metrics ============

def mean_reciprocal_rank(all_scores):
    """Compute Mean Reciprocal Rank (MRR)"""
    rr = []
    for scores in all_scores:
        # index 0 is the positive sample
        ranks = np.argsort(scores)[::-1]
        rank_pos = np.where(ranks == 0)[0][0] + 1
        rr.append(1.0 / rank_pos)
    return np.mean(rr)

def recall_at_k(all_scores, k=3):
    """Compute Recall@K"""
    hits = 0
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        if 0 in ranks[:k]:
            hits += 1
    return hits / len(all_scores)

def precision_at_k(all_scores, k=3):
    """Compute Precision@K"""
    precisions = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        top_k = ranks[:k]
        precisions.append(1 if 0 in top_k else 0)
    return np.mean(precisions)

def mean_average_precision(all_scores):
    """Compute Mean Average Precision (MAP)"""
    avg_precs = []
    for scores in all_scores:
        ranks = np.argsort(scores)[::-1]
        rank_pos = np.where(ranks == 0)[0][0] + 1
        avg_precs.append(1.0 / rank_pos)
    return np.mean(avg_precs)

# ============ Text Similarity Metrics ============

def evaluate_similarity(y_true, y_pred):
    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Pearson: {pearson:.4f} | Spearman: {spearman:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")

# ============ Main Baseline ============

def run_baseline(data):
    print("\n================= TEXT CLASSIFICATION =================")
    tc_train = data['train']['text_classification']
    tc_test = data['test']['text_classification']

    X_train, y_train = zip(*tc_train)
    X_test, y_test = zip(*tc_test)

    vectorizer = TfidfVectorizer(max_features=5000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds))

    # -----------------------------------------------------
    print("\n================= HEADLINE CLASSIFICATION =================")
    hc_train = data['train']['headline_classification']
    hc_test = data['test']['headline_classification']

    X_train, y_train = zip(*hc_train)
    X_test, y_test = zip(*hc_test)

    vectorizer = TfidfVectorizer(max_features=5000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds))

    # -----------------------------------------------------
    print("\n================= NLI =================")
    nli_train = data['train']['nli']
    nli_test = data['test']['nli']

    X_train = [a + " " + b for a, b, _ in nli_train]
    y_train = [label for _, _, label in nli_train]

    X_test = [a + " " + b for a, b, _ in nli_test]
    y_test = [label for _, _, label in nli_test]

    vectorizer = TfidfVectorizer(max_features=8000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds))

    # -----------------------------------------------------
    print("\n================= TEXT SIMILARITY (Regression) =================")
    sim_train = data['train']['similarity']
    sim_test = data['test']['similarity']

    X_train = [a + " " + b for a, b, _ in sim_train]
    y_train = [score for _, _, score in sim_train]

    X_test = [a + " " + b for a, b, _ in sim_test]
    y_test = [score for _, _, score in sim_test]

    vectorizer = TfidfVectorizer(max_features=8000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LinearRegression()
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    evaluate_similarity(y_test, preds)

    # -----------------------------------------------------
    print("\n================= RETRIEVAL =================")
    retrieval_train = data['train']['retrieval']
    retrieval_test = data['test']['retrieval']

    vectorizer = TfidfVectorizer(max_features=8000)
    corpus = []
    for q, p, negs in retrieval_train:
        corpus.extend([q, p] + negs)
    vectorizer.fit(corpus)

    all_scores = []
    for q, p, negs in retrieval_test:
        query_vec = vectorizer.transform([q])
        pos_vec = vectorizer.transform([p])
        neg_vecs = vectorizer.transform(negs)

        pos_score = (query_vec @ pos_vec.T).toarray()[0][0]
        neg_scores = (query_vec @ neg_vecs.T).toarray()[0]
        all_scores.append([pos_score] + neg_scores.tolist())

    mrr = mean_reciprocal_rank(all_scores)
    rec3 = recall_at_k(all_scores, k=1)
    prec3 = precision_at_k(all_scores, k=1)
    map_score = mean_average_precision(all_scores)

    print(f"MRR: {mrr:.4f} | Recall@1: {rec3:.4f} | Precision@1: {prec3:.4f} | MAP: {map_score:.4f}")

# ===============================================================
# Example: usage with your modified read_data()

def read_data():
    """Loads both train and test data (modify paths as needed)."""
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

# ===============================================================
if __name__ == "__main__":
    data = read_data()
    run_baseline(data)
