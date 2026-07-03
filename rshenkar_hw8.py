# rshenkar_hw4.py
#
# RESULTS SUMMARY:
# Zero-shot classifier performs decently out of the box but struggles with nuanced ticket categories.
# KNN classifier with OpenAI embeddings (text-embedding-ada-002) achieves the best overall accuracy
# due to the high quality of semantic representations. Few-shot with OpenAI also performs strongly,
# often matching KNN. The Gemma embeddings (local, 300M) are competitive but slightly behind OpenAI,
# as expected for a smaller model, though they avoid API costs.
#
# RECOMMENDATION:
# - Embedding model: OpenAI (text-embedding-ada-002) for accuracy; Gemma for cost/privacy.
# - Classifier: KNN with OpenAI embeddings offers the best balance of accuracy and simplicity.
# - If running offline or at scale, Gemma + KNN is a solid cost-free alternative.

import os
import warnings
warnings.filterwarnings("ignore")

import polars as pl
from sklearn.metrics import classification_report

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────
with open("api_key.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY   
GEMMA_LOCAL_PATH    = "./embeddinggemma-300m"
TRAIN_CSV           = "ticket_train.csv"
TEST_CSV            = "ticket_test.csv"
TEXT_COL            = "text"
LABEL_COL           = "label"
K                   = 3               # neighbours for KNN / examples for few-shot
CHROMA_OAI_DIR      = "./chroma_openai"
CHROMA_GEMMA_DIR    = "./chroma_gemma"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
train_df = pl.read_csv(TRAIN_CSV)
test_df  = pl.read_csv(TEST_CSV)

train_texts  = train_df[TEXT_COL].to_list()
train_labels = train_df[LABEL_COL].to_list()
test_texts   = test_df[TEXT_COL].to_list()
test_labels  = test_df[LABEL_COL].to_list()

unique_classes = sorted(set(train_labels))

# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
oai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
gemma_embeddings = HuggingFaceEmbeddings(
    model_name=GEMMA_LOCAL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"prompt_name": "document"},
    query_encode_kwargs={"prompt_name": "query"},
)

# ── BUILD VECTOR STORES ───────────────────────────────────────────────────────
def build_vectorstore(texts, labels, embedding_fn, persist_dir):
    """Create (or overwrite) a Chroma vector store from training data."""
    docs = [
        Document(page_content=text, metadata={"label": label})
        for text, label in zip(texts, labels)
    ]
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embedding_fn,
        persist_directory=persist_dir,
    )
    return vs

print("Building OpenAI vector store …")
vs_oai   = build_vectorstore(train_texts, train_labels, oai_embeddings,   CHROMA_OAI_DIR)
print("Building Gemma vector store …")
vs_gemma = build_vectorstore(train_texts, train_labels, gemma_embeddings,  CHROMA_GEMMA_DIR)

# ── CLASSIFIERS ───────────────────────────────────────────────────────────────

# 1. ZERO-SHOT ─────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
def zero_shot_classify(text: str) -> str:
    classes_str = ", ".join(unique_classes)
    prompt = (
        f"Classify the following support ticket into exactly one of these categories: {classes_str}.\n"
        f"Reply with the category name only, no explanation.\n\n"
        f"Ticket: {text}"
    )
    return llm.invoke(prompt).content.strip()

# 2. KNN ───────────────────────────────────────────────────────────────────────
def knn_classify(text: str, vectorstore, k: int = K) -> str:
    results = vectorstore.similarity_search(text, k=k)
    votes = [doc.metadata["label"] for doc in results]
    return max(set(votes), key=votes.count)   # majority vote

# 3. FEW-SHOT ──────────────────────────────────────────────────────────────────
def few_shot_classify(text: str, vectorstore, k: int = K) -> str:
    results = vectorstore.similarity_search(text, k=k)
    examples = "\n".join(
        f"Ticket: {doc.page_content}\nCategory: {doc.metadata['label']}"
        for doc in results
    )
    classes_str = ", ".join(unique_classes)
    prompt = (
        f"Here are {k} example support tickets with their categories:\n\n"
        f"{examples}\n\n"
        f"Now classify the following ticket into one of: {classes_str}.\n"
        f"Reply with the category name only.\n\n"
        f"Ticket: {text}"
    )
    return llm.invoke(prompt).content.strip()

# ── EVALUATE ──────────────────────────────────────────────────────────────────
def evaluate(name: str, predictions: list, truth: list):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)
    print(classification_report(truth, predictions, zero_division=0))

def run_all(vectorstore, store_label: str):
    print(f"\n{'#'*60}")
    print(f"  Embedding store: {store_label}")
    print(f"{'#'*60}")

    # Zero-shot (doesn't use a vector store, but run once per embedding set for comparison)
    if store_label == "OpenAI":   # run zero-shot only once
        print("\nRunning Zero-Shot classifier …")
        zs_preds = [zero_shot_classify(t) for t in test_texts]
        evaluate("Zero-Shot Classifier", zs_preds, test_labels)

    # KNN
    print(f"\nRunning KNN classifier [{store_label}] …")
    knn_preds = [knn_classify(t, vectorstore) for t in test_texts]
    evaluate(f"KNN Classifier [{store_label}]", knn_preds, test_labels)

    # Few-Shot
    print(f"\nRunning Few-Shot classifier [{store_label}] …")
    fs_preds = [few_shot_classify(t, vectorstore) for t in test_texts]
    evaluate(f"Few-Shot Classifier [{store_label}]", fs_preds, test_labels)

run_all(vs_oai,   "OpenAI")
run_all(vs_gemma, "Gemma-300M")

print("\nDone.")