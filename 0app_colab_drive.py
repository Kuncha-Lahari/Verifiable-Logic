"""
Verifiable Logic: Enhancing and Auditing the Faithfulness of LLM Decision-Making
through Targeted Fine-Tuning and XAI
===================================================================================

DUAL-MODEL ARCHITECTURE
  Model 1 — Prediction Engine  : GradientBoostingClassifier + SHAP TreeExplainer
  Model 2 — Explanation Engine : Flan-T5-Small + LoRA (PEFT) fine-tuning

PIPELINE
  1. Model 1 predicts ACCEPT/REJECT and computes SHAP attributions
  2. SHAP values embedded into a structured prompt (verifiable constraint)
  3. Fine-tuned Flan-T5 generates CoT explanation from that prompt
  4. Faithfulness scorer checks explanation polarity vs SHAP sign (0-1)
  5. Self-Refine loop: if score < threshold, LLM regenerates with correction
  6. Hard post-correction: remaining wrong-direction sentences rewritten
  7. Hallucination detector flags Type1 (ghost) and Type2 (polarity) errors
  8. Everything shown in real-time Streamlit dashboard
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import re
import os
import warnings

import torch
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — avoids threading issues
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Verifiable Logic — XAI Dashboard", layout="wide")

# ── Persistent paths: save models/results to Google Drive in Colab ──────────────
# Falls back to the current working directory when Drive is unavailable.
if os.path.isdir("/content/drive/MyDrive"):
    PROJECT_DIR = "/content/drive/MyDrive/Verifiable_Logic"
else:
    PROJECT_DIR = os.path.abspath("verifiable_logic_project")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
LOGS_DIR    = os.path.join(PROJECT_DIR, "logs")
LORA_DIR   = os.path.join(MODELS_DIR, "verifiable_logic_lora")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df        = pd.read_csv("law_data.csv")
    desired   = ["ugpa", "lsat", "decile1b", "decile3", "fam_inc", "cluster", "pass_bar"]
    available = [c for c in desired if c in df.columns]
    df        = df[available].dropna()
    if "pass_bar" not in df.columns:
        st.error("Column 'pass_bar' not found in CSV.")
        st.stop()
    # Keep ALL rows — class imbalance is handled by sample_weight in the model.
    # Undersampling would discard up to 75% of the dataset, hurting accuracy.
    df = df.reset_index(drop=True)
    return df, [c for c in available if c != "pass_bar"]

df, FEATURE_COLS = load_data()
X_all = df[FEATURE_COLS]
y_all = df["pass_bar"]

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MODEL 1: PREDICTION ENGINE
# Underscore-prefixed args (_X, _y) tell Streamlit not to hash DataFrames
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_prediction_model(_X, _y):
    # Stratified split: preserves the class ratio in both train and test sets.
    # This is critical when the dataset is imbalanced (e.g. ~75% ACCEPT).
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )
    scaler  = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te_sc = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns)

    # Re-index so iloc[i] and positional access always agree
    X_tr_sc = X_tr_sc.reset_index(drop=True)
    X_te_sc = X_te_sc.reset_index(drop=True)

    # compute_sample_weight assigns higher weight to the minority class.
    # This handles class imbalance without discarding any data.
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)

    # Tuned GradientBoostingClassifier:
    #   - 500 trees with lower learning_rate (0.05) → better generalisation
    #   - subsample=0.85 → stochastic GBM, reduces overfitting
    #   - max_depth=5    → captures higher-order interactions in LSAC features
    #   - min_samples_leaf=4 → prevents overfitting on small leaf nodes
    #   - max_features='sqrt' → feature subsampling per tree (like RF), reduces variance
    #   - n_iter_no_change=30 → early stopping if no improvement for 30 rounds
    #   - validation_fraction=0.1 → internal val set used for early stopping
    mdl = GradientBoostingClassifier(
        n_estimators       = 500,
        learning_rate      = 0.05,
        max_depth          = 5,
        subsample          = 0.85,
        min_samples_leaf   = 4,
        max_features       = "sqrt",
        validation_fraction= 0.1,
        n_iter_no_change   = 30,
        tol                = 1e-4,
        random_state       = 42,
    )
    mdl.fit(X_tr_sc, y_tr, sample_weight=sample_weights)
    expl = shap.TreeExplainer(mdl)

    # Store original (unscaled) test rows for feature-value display
    X_te_orig = X_te.reset_index(drop=True)

    return mdl, scaler, expl, X_tr_sc, X_te_sc, y_te.reset_index(drop=True), X_te_orig

model, scaler, explainer, X_train_sc, X_test_sc, y_test, X_test_orig = \
    train_prediction_model(X_all, y_all)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — FAITHFULNESS UTILITIES  (defined first — used everywhere below)
# ══════════════════════════════════════════════════════════════════════════════

def _direction(sv: float) -> str:
    """Correct SHAP-grounded direction phrase for a given SHAP value."""
    if sv < -1.0:  return "very strongly pushes toward REJECT"
    elif sv < 0:   return "pushes toward REJECT"
    elif sv > 1.0: return "very strongly pushes toward ACCEPT"
    else:          return "pushes toward ACCEPT"

_ACCEPT_KW = [
    "pushes toward accept", "supports accept", "pushes accept",
    "positive effect", "toward accept", "in favor",
    "pushes to accept", "leads to accept", "very strongly pushes toward accept",
]
_REJECT_KW = [
    "pushes toward reject", "supports reject", "pushes reject",
    "negative effect", "hurts", "toward reject",
    "leads to reject", "pushes to reject", "very strongly pushes toward reject",
]

def _split_sentences(text: str) -> list:
    """
    Split on '. ' only when NOT preceded by a digit.
    Prevents decimal numbers like -3.139 splitting into separate fragments.
    """
    if not text:
        return []
    parts = re.split(r'(?<!\d)(?<!\d\d)\.(?=\s+[A-Z])', text)
    return [s.strip().rstrip('.') for s in parts if s.strip()]

def _polarity(sentence: str) -> str:
    """Returns 'positive', 'negative', or 'unclear' for one sentence."""
    s = sentence.lower()
    if any(kw in s for kw in _ACCEPT_KW): return "positive"
    if any(kw in s for kw in _REJECT_KW): return "negative"
    return "unclear"

def compute_faithfulness(explanation: str, shap_dict: dict, top_n: int = 3):
    """
    Score = fraction of top-N SHAP features whose implied direction
    in the LLM explanation matches the actual SHAP sign.
    Returns (score: float, details: list[tuple]).
    """
    if not explanation or not shap_dict:
        return 0.0, []

    top_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sentences = _split_sentences(explanation)
    score, details = 0, []

    for feat, sv in top_feats:
        shap_sign  = "negative" if sv < 0 else "positive"
        feat_sents = [s for s in sentences if feat.lower() in s.lower()]

        if not feat_sents:
            details.append((feat, sv, "❌ No",
                            f"'{feat}' not mentioned (SHAP={sv:.3f}) — omission"))
            continue

        polarities = [_polarity(s) for s in feat_sents]
        implied    = next((p for p in polarities if p != "unclear"), "unclear")

        if implied == shap_sign:
            score += 1
            details.append((feat, sv, "✅ Correct",
                            f"LLM correctly states '{feat}' is {shap_sign}"))
        elif implied == "unclear":
            details.append((feat, sv, "⚠️ Unclear",
                            f"LLM mentions '{feat}' but direction is ambiguous"))
        else:
            details.append((feat, sv, "❌ No",
                            f"LLM says {implied} for '{feat}', SHAP says {shap_sign} ({sv:.3f})"))

    return round(score / max(top_n, 1), 2), details

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SYNTHETIC CoT FINE-TUNING DATASET
# SHAP-grounded: every (prompt, target) pair is mathematically faithful.
# Teaching the LLM faithfulness by example — a research strength.
# ══════════════════════════════════════════════════════════════════════════════
def make_prompt(feat_vals: dict, shap_vals: dict,
                decision: str, confidence: float) -> str:
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = "\n".join(
        f"  {f}: value={feat_vals[f]:.1f}, SHAP={sv:.3f} ({_direction(sv)})"
        for f, sv in sorted_feats
    )
    return (
        f"You are an XAI auditor explaining law school admission decisions.\n"
        f"Given SHAP feature attributions, write a faithful chain-of-thought "
        f"explanation.\n\nFeatures and SHAP values:\n{lines}\n\n"
        f"Decision: {decision} (confidence: {round(confidence * 100, 1)}%)\n\n"
        f"Explanation:"
    )

def make_target(feat_vals: dict, shap_vals: dict, decision: str) -> str:
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    sentences = []
    for i, (f, sv) in enumerate(sorted_feats):
        d = _direction(sv)
        if i == 0:
            sentences.append(
                f"The most influential factor is {f} "
                f"(value: {feat_vals[f]:.1f}), which {d} "
                f"with a SHAP contribution of {sv:.3f}."
            )
        else:
            sentences.append(
                f"Their {f} (value: {feat_vals[f]:.1f}) also {d} "
                f"(SHAP: {sv:.3f})."
            )
    sentences.append(
        f"Overall, the combined profile leads to a {decision} decision."
    )
    return " ".join(sentences)

def build_finetune_dataset(X_data, shap_matrix, predictions, probabilities, n=300):
    records = []
    for i in range(min(n, len(X_data))):
        row       = X_data.iloc[i]
        shap_row  = dict(zip(FEATURE_COLS, shap_matrix[i]))
        feat_vals = {f: float(row[f]) for f in FEATURE_COLS}
        pred      = int(predictions[i])
        prob      = float(probabilities[i][pred])
        decision  = "ACCEPT" if pred == 1 else "REJECT"
        records.append({
            "input":  make_prompt(feat_vals, shap_row, decision, prob),
            "target": make_target(feat_vals, shap_row, decision),
        })
    return records

class CoTDataset(Dataset):
    """
    PyTorch Dataset for Seq2Seq fine-tuning.
    Uses text_target= parameter (transformers >= 4.32 API).
    Falls back to plain tokenizer call if text_target is unsupported.
    """
    def __init__(self, records, tokenizer, max_input=256, max_target=150):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Encode source
        enc = self.tokenizer(
            rec["input"],
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Encode target — use text_target= (modern API, no as_target_tokenizer)
        # If this raises, fall back to plain call (some older builds)
        try:
            lbl_enc = self.tokenizer(
                text_target=rec["target"],
                max_length=self.max_target,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        except TypeError:
            # Fallback for edge cases: tokenize target as plain text
            lbl_enc = self.tokenizer(
                rec["target"],
                max_length=self.max_target,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        # .clone() avoids in-place modification of a leaf tensor (PyTorch bug)
        labels = lbl_enc["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — MODEL 2: FLAN-T5 + LoRA FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_finetune_llm(_X_train, _explainer, _model):
    """
    1. Load Flan-T5-small + AutoTokenizer
    2. Attach LoRA adapter (r=8, alpha=16, targets q & v projections)
    3. Build 300 SHAP-grounded synthetic CoT training examples
    4. Fine-tune 3 epochs with Seq2SeqTrainer
    5. Return (tokenizer, eval-mode peft model)
    """
    base_name = "google/flan-t5-small"

    # Reuse the trained LoRA adapter from Drive when it already exists.
    # This prevents Streamlit reruns from retraining the model.
    adapter_config = os.path.join(LORA_DIR, "adapter_config.json")
    if os.path.exists(adapter_config):
        tok = AutoTokenizer.from_pretrained(LORA_DIR)
        base_mdl = T5ForConditionalGeneration.from_pretrained(base_name)
        peft_mdl = PeftModel.from_pretrained(base_mdl, LORA_DIR)
        peft_mdl = peft_mdl.to(DEVICE)
        peft_mdl.eval()

        loss_path = os.path.join(LORA_DIR, "training_loss.json")
        if os.path.exists(loss_path):
            import json
            with open(loss_path, "r", encoding="utf-8") as f:
                train_loss_log = json.load(f)
        else:
            train_loss_log = []
        return tok, peft_mdl, train_loss_log

    tok       = AutoTokenizer.from_pretrained(base_name)
    base_mdl  = T5ForConditionalGeneration.from_pretrained(base_name)

    lora_cfg = LoraConfig(
        task_type      = TaskType.SEQ_2_SEQ_LM,
        r              = 8,
        lora_alpha     = 16,
        lora_dropout   = 0.1,
        target_modules = ["q", "v"],
    )
    peft_mdl = get_peft_model(base_mdl, lora_cfg)

    # Build dataset — handle both list-of-2-arrays and single 2-D array
    # SHAP binary classification can return either depending on version
    shap_raw = _explainer.shap_values(_X_train)
    if isinstance(shap_raw, list):
        # Older SHAP: list of [neg_class_arr, pos_class_arr]
        shap_mat = shap_raw[1]
    elif shap_raw.ndim == 3:
        # Some versions: shape (n_samples, n_features, n_classes)
        shap_mat = shap_raw[:, :, 1]
    else:
        # Newer SHAP: single 2-D array (n_samples, n_features)
        shap_mat = shap_raw

    preds   = _model.predict(_X_train)
    probs   = _model.predict_proba(_X_train)
    records = build_finetune_dataset(_X_train, shap_mat, preds, probs, n=300)
    dataset = CoTDataset(records, tok)

    train_args = Seq2SeqTrainingArguments(
        output_dir                  = LORA_DIR,
        num_train_epochs            = 3,
        per_device_train_batch_size = 4,
        learning_rate               = 3e-4,
        save_strategy               = "no",
        logging_strategy            = "steps",  # capture loss every N steps
        logging_steps               = 10,        # record every 10 steps
        predict_with_generate       = False,     # no eval set → disable to avoid warning
        fp16                        = False,
        report_to                   = "none",
    )
    collator = DataCollatorForSeq2Seq(tok, model=peft_mdl, padding=True)
    trainer  = Seq2SeqTrainer(
        model         = peft_mdl,
        args          = train_args,
        train_dataset = dataset,
        data_collator = collator,
    )
    trainer.train()
    peft_mdl.eval()

    # Persist the adapter and tokenizer to Google Drive.
    os.makedirs(LORA_DIR, exist_ok=True)
    peft_mdl.save_pretrained(LORA_DIR)
    tok.save_pretrained(LORA_DIR)

    # Extract training loss history for the loss curve plot
    train_loss_log = [
        {"step": x["step"], "loss": x["loss"]}
        for x in trainer.state.log_history
        if "loss" in x
    ]

    import json
    with open(os.path.join(LORA_DIR, "training_loss.json"), "w", encoding="utf-8") as f:
        json.dump(train_loss_log, f, indent=2)

    peft_mdl = peft_mdl.to(DEVICE)
    return tok, peft_mdl, train_loss_log

tokenizer, llm_model, TRAIN_LOSS_LOG = load_and_finetune_llm(X_train_sc, explainer, model)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — CoT GENERATION + SELF-REFINE + HARD POST-CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
MAX_REFINE_ROUNDS = 3
REFINE_THRESHOLD  = 0.67


def _run_llm(prompt: str) -> tuple:
    """
    Single Flan-T5 inference call.
    Returns (generated_text, token_scores).
    token_scores = list of (token_str, real_softmax_prob).

    T5 seq2seq: outputs.sequences starts from decoder_start_token_id.
    outputs.scores has one entry per *generated* token (after start token).
    So scores[i] corresponds to sequences[i+1] (offset by 1 for start token).
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens          = 150,
            num_beams               = 4,
            early_stopping          = True,
            return_dict_in_generate = True,
            output_scores           = True,
        )

    # Decode generated text
    # sequences[0] = full sequence including decoder start token
    # skip_special_tokens removes pad, eos etc.
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()

    # Token-level confidence:
    # outputs.scores[i] = logits over vocab for step i (0-indexed from first real token)
    # outputs.sequences[0][i+1] = the token chosen at step i (offset by start token)
    token_scores = []
    seq = outputs.sequences[0]
    for step, score_tensor in enumerate(outputs.scores):
        seq_idx  = step + 1          # +1 to skip the decoder start token
        if seq_idx >= len(seq):
            break
        token_id = seq[seq_idx].item()
        probs    = torch.softmax(score_tensor[0], dim=-1)   # [0] = first beam
        tok_prob = probs[token_id].item()
        tok_str  = tokenizer.decode([token_id])
        if tok_str.strip() and tok_str not in ("<pad>", "</s>", "<unk>"):
            token_scores.append((tok_str, round(tok_prob, 4)))

    return text, token_scores


def _build_refine_prompt(original: str, failing: list, decision: str) -> str:
    """Targeted correction prompt listing exactly what the LLM got wrong."""
    corrections = "\n".join(
        f"  - '{feat}': stated {wrong_dir} effect, but "
        f"SHAP={sv:.3f} means it {_direction(sv)}. Please correct."
        for feat, sv, wrong_dir in failing
    )
    return (
        f"The following explanation has faithfulness errors:\n\n"
        f"Original:\n{original}\n\n"
        f"Required corrections:\n{corrections}\n\n"
        f"Decision: {decision}\n\n"
        f"Rewrite so every feature direction matches its SHAP value. "
        f"Corrected explanation:"
    )


def _hard_post_correct(explanation: str, shap_dict: dict) -> str:
    """
    Hard Constraint Post-Correction.
    Any sentence whose polarity contradicts SHAP is surgically rewritten.
    Converts soft prompt constraint into a hard output guarantee.
    """
    if not explanation:
        return explanation
    sentences = _split_sentences(explanation)
    corrected = []

    for sent in sentences:
        modified = sent
        for feat, sv in shap_dict.items():
            if feat.lower() not in sent.lower():
                continue
            implied   = _polarity(sent)
            shap_sign = "negative" if sv < 0 else "positive"
            if implied == "unclear" or implied == shap_sign:
                break   # this sentence is fine
            # Replace the wrong direction phrase with the correct one
            correct_phrase = _direction(sv)
            wrong_kws      = _REJECT_KW if shap_sign == "positive" else _ACCEPT_KW
            for wrong in sorted(wrong_kws, key=len, reverse=True):  # longest first
                if wrong in modified.lower():
                    modified = re.sub(
                        re.escape(wrong), correct_phrase,
                        modified, flags=re.IGNORECASE, count=1
                    )
                    break
            break
        corrected.append(modified)

    return ". ".join(corrected)


def generate_explanation(shap_dict: dict, feat_vals: dict,
                         prediction: int, confidence: float) -> tuple:
    """
    Full pipeline: A. Generate → B. Self-Refine → C. Hard post-correct.
    Returns (final_explanation, token_scores, refine_history).
    """
    decision       = "ACCEPT" if prediction == 1 else "REJECT"
    prompt         = make_prompt(feat_vals, shap_dict, decision, confidence)
    refine_history = []

    # A — initial generation
    explanation, token_scores = _run_llm(prompt)
    score, details            = compute_faithfulness(explanation, shap_dict)
    refine_history.append({
        "round": 0, "label": "Initial generation",
        "explanation": explanation, "score": score,
    })

    # B — Self-Refine loop
    for rnd in range(1, MAX_REFINE_ROUNDS + 1):
        if score >= REFINE_THRESHOLD:
            break
        failing = [
            (feat, sv, "positive" if sv < 0 else "negative")
            for feat, sv, verdict, _ in details
            if verdict == "❌ No"
        ]
        if not failing:
            break   # only unclear — more rounds won't help
        refine_prompt             = _build_refine_prompt(explanation, failing, decision)
        explanation, token_scores = _run_llm(refine_prompt)
        score, details            = compute_faithfulness(explanation, shap_dict)
        refine_history.append({
            "round": rnd, "label": f"Self-Refine round {rnd}",
            "explanation": explanation, "score": score,
        })

    # C — Hard constraint post-correction
    corrected = _hard_post_correct(explanation, shap_dict)
    if corrected != explanation:
        explanation    = corrected
        score, details = compute_faithfulness(explanation, shap_dict)
        refine_history.append({
            "round": "HC", "label": "Hard constraint post-correction",
            "explanation": explanation, "score": score,
        })

    return explanation, token_scores, refine_history

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — HALLUCINATION DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_hallucinations(explanation: str, shap_dict: dict) -> list:
    """
    Type 1 — Ghost: LLM over-emphasises a feature with low relative SHAP
    Type 2 — Polarity: LLM states wrong direction for a feature vs SHAP sign
    """
    if not explanation or not shap_dict:
        return []

    hallucinations = []
    sorted_feats   = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    sentences      = _split_sentences(explanation)
    max_shap       = max(abs(v) for v in shap_dict.values())
    if max_shap == 0:
        return []

    for feat, sv in sorted_feats:
        feat_sents = [s for s in sentences if feat.lower() in s.lower()]
        if not feat_sents:
            continue

        # Type 2
        for sent in feat_sents:
            implied   = _polarity(sent)
            shap_sign = "negative" if sv < 0 else "positive"
            if implied != "unclear" and implied != shap_sign:
                hallucinations.append({
                    "type": "Type 2 — Polarity contradiction",
                    "message": (
                        f"LLM implies '{feat}' has a {implied} effect, "
                        f"but SHAP={sv:.3f} ({shap_sign}). Contradicts the math."
                    ),
                })

        # Type 1
        rel_imp  = abs(sv) / max_shap
        emphasis = [
            "most influential", "decisive", "critical", "primarily",
            "main reason", "key factor", "most important", "strongest",
        ]
        if rel_imp < 0.15:
            for sent in feat_sents:
                if any(w in sent.lower() for w in emphasis):
                    hallucinations.append({
                        "type": "Type 1 — Ghost feature",
                        "message": (
                            f"LLM over-emphasises '{feat}', but its relative "
                            f"SHAP importance is only {round(rel_imp*100,1)}% "
                            f"of the top feature — fabricated importance."
                        ),
                    })
                    break

    return hallucinations

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔍 Verifiable Logic — Faithfulness Dashboard")
st.caption("Real-time auditing of LLM explanations against SHAP feature attributions.")
st.caption(f"Runtime device: {DEVICE}  |  Persistent project: {PROJECT_DIR}")

with st.expander("⚙️ System Architecture", expanded=False):
    a1, a2, a3 = st.columns(3)
    a1.info(
        "**Model 1 — Prediction Engine**\n\n"
        "GradientBoostingClassifier\n\n"
        "500 trees · lr=0.05 · depth=5\n\n"
        "→ ACCEPT / REJECT + confidence\n\n"
        "→ SHAP TreeExplainer"
    )
    a2.success(
        "**Model 2 — Explanation Engine**\n\n"
        "Flan-T5-Small (Google)\n\n"
        "→ LoRA fine-tuned (PEFT)\n\n"
        "→ 300 SHAP-grounded CoT examples\n\n"
        "→ SHAP-constrained prompt"
    )
    a3.warning(
        "**Auditing Pipeline**\n\n"
        "① Faithfulness Scoring\n\n"
        "② Self-Refine (≤3 rounds)\n\n"
        "③ Hard Constraint Correction\n\n"
        "④ Hallucination Detection"
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Select Test Sample")
max_idx    = len(X_test_sc) - 1
sample_idx = st.sidebar.slider("Sample index", 0, max_idx, min(100, max_idx))
st.sidebar.markdown("---")
st.sidebar.subheader("Or Enter Custom Values")
use_custom = st.sidebar.checkbox("Use custom applicant values")

if use_custom:
    custom_vals = {
        feat: st.sidebar.slider(
            feat,
            float(X_all[feat].min()),
            float(X_all[feat].max()),
            float(X_all[feat].mean()),
        )
        for feat in FEATURE_COLS
    }
    input_raw = pd.DataFrame([custom_vals])
else:
    input_raw = X_test_sc.iloc[[sample_idx]].copy()

# ── Model 1: Prediction + SHAP ────────────────────────────────────────────────
if use_custom:
    input_sc = pd.DataFrame(
        scaler.transform(input_raw), columns=FEATURE_COLS
    )
else:
    input_sc = input_raw

pred = int(model.predict(input_sc)[0])
prob = float(model.predict_proba(input_sc)[0][pred])

# SHAP: handle all output shapes robustly
shap_raw = explainer.shap_values(input_sc)
if isinstance(shap_raw, list):
    sv_arr = shap_raw[1][0]
elif np.array(shap_raw).ndim == 3:
    sv_arr = np.array(shap_raw)[0, :, 1]
else:
    sv_arr = np.array(shap_raw)[0]

shap_dict = dict(zip(FEATURE_COLS, sv_arr))

# Feature values (original unscaled) for display and prompts
if use_custom:
    feat_vals = {f: float(custom_vals[f]) for f in FEATURE_COLS}
else:
    # X_test_orig is aligned by reset index with X_test_sc
    feat_vals = {f: float(X_test_orig.iloc[sample_idx][f]) for f in FEATURE_COLS}

# ── Model 2: generate + self-refine + hard post-correct ───────────────────────
with st.spinner("🧠 Flan-T5 generating Chain-of-Thought (self-refine active)..."):
    cot_explanation, token_scores, refine_history = generate_explanation(
        shap_dict, feat_vals, pred, prob
    )

# ── Faithfulness + hallucination ─────────────────────────────────────────────
faith_score, faith_details = compute_faithfulness(cot_explanation, shap_dict)
hallucinations             = detect_hallucinations(cot_explanation, shap_dict)
hallucination_detected     = len(hallucinations) > 0

# Persist the current audited sample to Drive.
try:
    import json
    sample_result = {
        "sample_index": int(sample_idx),
        "custom_input": bool(use_custom),
        "prediction": "ACCEPT" if pred == 1 else "REJECT",
        "confidence": float(prob),
        "shap_values": {k: float(v) for k, v in shap_dict.items()},
        "explanation": cot_explanation,
        "faithfulness_score": float(faith_score),
        "faithfulness_details": [list(x) for x in faith_details],
        "hallucinations": hallucinations,
        "refine_history": refine_history,
    }
    with open(os.path.join(RESULTS_DIR, f"sample_{sample_idx}_audit.json"), "w", encoding="utf-8") as f:
        json.dump(sample_result, f, indent=2, ensure_ascii=False)
except Exception:
    pass

# ── Metrics row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Prediction",  "✅ ACCEPT" if pred == 1 else "❌ REJECT")
m2.metric("Confidence",  f"{round(prob * 100, 1)}%")

faith_emoji = "✅" if faith_score == 1.0 else ("⚠️" if faith_score >= 0.5 else "❌")
m3.metric("Faithfulness Score", f"{faith_score} / 1.0  {faith_emoji}")

hall_label = (
    "✅ None"     if not hallucination_detected else
    "⚠️ Maybe"   if faith_score >= 0.5         else
    "🚨 Detected"
)
m4.metric("Hallucination", hall_label)

st.divider()

# ── SHAP chart + LLM explanation ─────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("📊 SHAP Feature Importance  _(Model 1)_")
    s_sorted = sorted(shap_dict.items(), key=lambda x: x[1])
    feats    = [s[0] for s in s_sorted]
    vals     = [s[1] for s in s_sorted]
    colors   = ["#d73027" if v < 0 else "#1a9850" for v in vals]
    fig, ax  = plt.subplots(figsize=(6, max(3, len(feats) * 0.6 + 1)))
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Feature Contributions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("🧠 LLM Chain-of-Thought  _(Flan-T5 + LoRA)_")
    if cot_explanation:
        st.info(cot_explanation)
    else:
        st.warning("Model produced no output. Try a different sample index.")

    st.markdown("**Token-Level Confidence** _(real Flan-T5 softmax logit probabilities)_")
    if token_scores:
        html_parts = []
        for tok_str, p in token_scores:
            intensity = min(1.0, max(0.2, p * 3))
            color     = (
                f"rgba(220,38,38,{intensity:.2f})" if pred == 0
                else f"rgba(22,163,74,{intensity:.2f})"
            )
            safe_tok = (tok_str
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            html_parts.append(
                f'<span title="p={p}" style="background:{color};padding:2px 4px;'
                f'border-radius:3px;margin:1px;display:inline-block;'
                f'font-size:13px;">{safe_tok}</span>'
            )
        st.markdown("".join(html_parts), unsafe_allow_html=True)
        st.caption(
            "Colour intensity = real softmax probability from Flan-T5 decoder logits. "
            "Brighter = model was more confident about that token."
        )
    else:
        st.caption("No token scores available.")

st.divider()

# ── Faithfulness audit table ──────────────────────────────────────────────────
st.subheader("📋 Faithfulness Audit — Feature-by-Feature")
if faith_details:
    rows = []
    for feat, sv, verdict, reason in faith_details:
        rows.append({
            "Feature":        feat,
            "SHAP Value":     round(sv, 3),
            "SHAP Direction": "↑ Pushes ACCEPT" if sv > 0 else "↓ Pushes REJECT",
            "In LLM text?":   "Yes" if feat.lower() in cot_explanation.lower() else "No",
            "Faithful?":      verdict,
            "Reason":         reason,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No audit details available.")

st.divider()

# ── Self-Refine history ───────────────────────────────────────────────────────
st.subheader("🔄 Self-Refine Loop History")
if len(refine_history) <= 1:
    st.success("✅ Initial generation was faithful — no refinement needed.")
else:
    st.info(
        f"Pipeline ran **{len(refine_history) - 1}** correction step(s). "
        f"Final faithfulness: **{faith_score}**"
    )

refine_rows = []
for entry in refine_history:
    preview = entry["explanation"]
    if len(preview) > 110:
        preview = preview[:110] + "..."
    refine_rows.append({
        "Step":               entry["label"],
        "Faithfulness Score": entry["score"],
        "Explanation preview":preview,
    })
st.dataframe(pd.DataFrame(refine_rows), use_container_width=True)

if len(refine_history) > 1:
    scores = [e["score"] for e in refine_history]
    labels = [str(e["round"]) for e in refine_history]
    fig2, ax2 = plt.subplots(figsize=(5, 2.5))
    ax2.plot(labels, scores, marker="o", color="#1a9850", linewidth=2)
    ax2.axhline(REFINE_THRESHOLD, color="#d73027", linestyle="--",
                linewidth=1, label=f"Threshold ({REFINE_THRESHOLD})")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Faithfulness score")
    ax2.set_title("Faithfulness across self-refine rounds")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ── Hallucination panel ───────────────────────────────────────────────────────
st.subheader("🚨 Hallucination Detection  _(real Flan-T5 output — no injection)_")
if hallucinations:
    for h in hallucinations:
        st.error(f"**{h['type']}**\n\n{h['message']}")
    top1 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[0]
    st.warning(
        f"#1 SHAP feature: **{top1[0]}** (SHAP={top1[1]:.3f}). "
        f"See audit table for full breakdown."
    )
else:
    st.success(
        "✅ No hallucinations detected. "
        "No Type 1 or Type 2 issue was detected by the implemented audit rules."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6.2 — PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
st.header("📈 Section 6.2 — Performance Evaluation")

# ── (A) Training Loss Curve ───────────────────────────────────────────────────
st.subheader("(A) Training Loss Curve  _(LoRA Fine-Tuning of Flan-T5)_")
if TRAIN_LOSS_LOG:
    steps      = [x["step"] for x in TRAIN_LOSS_LOG]
    loss_vals  = [x["loss"] for x in TRAIN_LOSS_LOG]
    fig_loss, ax_loss = plt.subplots(figsize=(7, 3))
    ax_loss.plot(steps, loss_vals, color="#1565C0", linewidth=2, marker="o",
                 markersize=3, label="Training loss")
    ax_loss.set_title("Training Loss Curve — LoRA Fine-Tuning", fontsize=13)
    ax_loss.set_xlabel("Training Steps")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    st.pyplot(fig_loss)
    plt.close(fig_loss)
    st.caption(
        "The curve reports optimization loss during LoRA fine-tuning on the "
        "SHAP-grounded training examples; lower training loss indicates better fit "
        "to the training objective."
    )
else:
    st.info("Training loss log is empty — re-run the app to populate it.")

st.divider()

# ── (B) Model Accuracy + Faithfulness Distribution ────────────────────────────
st.subheader("(B) Accuracy & Faithfulness Distribution")

# Model 1 accuracy (classification)
y_pred_all = model.predict(X_test_sc)
acc_val    = accuracy_score(y_test, y_pred_all)

b1, b2, b3 = st.columns(3)
b1.metric("Model 1 Accuracy",   f"{round(acc_val * 100, 1)}%")
b2.metric("Avg Confidence",     f"{round(float(model.predict_proba(X_test_sc)[:, 1].mean()) * 100, 1)}%")
b3.metric("Total Test Samples", len(X_test_sc))

# Faithfulness distribution across ALL test samples (fast — no LLM, uses stored shap)
st.markdown("**Faithfulness Score Distribution (across test set)**")
st.caption("Computed on the first 50 samples using the full generation pipeline.")

with st.spinner("Computing faithfulness scores across 50 test samples..."):
    @st.cache_data(show_spinner=False)
    def compute_faith_distribution(_model, _explainer, _X_te, _X_te_orig, _feat_cols, n=50):
        scores = []
        for i in range(min(n, len(_X_te))):
            row  = _X_te.iloc[[i]]
            p    = int(_model.predict(row)[0])
            prob = float(_model.predict_proba(row)[0][p])
            shap_raw = _explainer.shap_values(row)
            if isinstance(shap_raw, list):
                sv = shap_raw[1][0]
            elif hasattr(shap_raw, "ndim") and shap_raw.ndim == 3:
                sv = shap_raw[0, :, 1]
            else:
                sv = shap_raw[0]
            sd = dict(zip(_feat_cols, sv))
            fv = {f: float(_X_te_orig.iloc[i][f]) for f in _feat_cols}
            exp, _, _ = generate_explanation(sd, fv, p, prob)
            sc, _     = compute_faithfulness(exp, sd)
            scores.append(sc)
        return scores

    faith_dist = compute_faith_distribution(
        model, explainer, X_test_sc, X_test_orig, FEATURE_COLS
    )

fig_faith, axes_faith = plt.subplots(1, 2, figsize=(10, 3.5))

# Line plot
axes_faith[0].plot(faith_dist, color="#2DC653", linewidth=1.5)
axes_faith[0].axhline(REFINE_THRESHOLD, color="#d73027", linestyle="--",
                       linewidth=1, label=f"Threshold ({REFINE_THRESHOLD})")
axes_faith[0].set_ylim(0, 1.05)
axes_faith[0].set_title("Faithfulness Score per Sample", fontsize=11)
axes_faith[0].set_xlabel("Sample Index")
axes_faith[0].set_ylabel("Faithfulness Score")
axes_faith[0].legend(fontsize=9)
axes_faith[0].spines["top"].set_visible(False)
axes_faith[0].spines["right"].set_visible(False)

# Histogram
axes_faith[1].hist(faith_dist, bins=10, color="#1565C0", edgecolor="white", alpha=0.85)
axes_faith[1].set_title("Faithfulness Score Distribution", fontsize=11)
axes_faith[1].set_xlabel("Faithfulness Score")
axes_faith[1].set_ylabel("Count")
axes_faith[1].spines["top"].set_visible(False)
axes_faith[1].spines["right"].set_visible(False)

plt.tight_layout()
st.pyplot(fig_faith)
plt.close(fig_faith)

mean_faith = round(sum(faith_dist) / len(faith_dist), 3) if faith_dist else 0
st.caption(
    f"Mean faithfulness score: **{mean_faith}** across {len(faith_dist)} samples. "
    "This score summarizes agreement between the generated explanation and the "
    "SHAP-direction checks for the evaluated samples."
)

st.divider()

# ── (C) Confusion Matrix ──────────────────────────────────────────────────────
st.subheader("(C) Confusion Matrix  _(Model 1: GradientBoostingClassifier)_")

cm = confusion_matrix(y_test, y_pred_all)
fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))

if _HAS_SNS:
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
        xticklabels=["REJECT", "ACCEPT"],
        yticklabels=["REJECT", "ACCEPT"],
        annot_kws={"size": 14, "weight": "bold"},
    )
else:
    # Fallback: manual heatmap without seaborn
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax_cm)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["REJECT", "ACCEPT"])
    ax_cm.set_yticklabels(["REJECT", "ACCEPT"])

ax_cm.set_title("Confusion Matrix — Model 1 Predictions", fontsize=12)
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
plt.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)

tn, fp, fn, tp = cm.ravel()
precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0, 3)
recall    = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 3)
f1        = round(2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0, 3)

cm1, cm2, cm3, cm4 = st.columns(4)
cm1.metric("Accuracy",  f"{round(acc_val * 100, 1)}%")
cm2.metric("Precision", f"{precision}")
cm3.metric("Recall",    f"{recall}")
cm4.metric("F1-Score",  f"{f1}")

try:
    metrics = {
        "model_1_accuracy": float(acc_val),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "test_samples": int(len(X_test_sc)),
        "mean_faithfulness_first_50": float(mean_faith) if "mean_faith" in globals() else None,
    }
    import json
    with open(os.path.join(RESULTS_DIR, "performance_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
except Exception:
    pass

st.caption(
    "The confusion matrix reports classification performance on the held-out test set."
)

# ── Mean SHAP importance ──────────────────────────────────────────────────────
st.divider()
st.subheader("(D) Mean |SHAP| Feature Importance  _(across all test samples)_")

shap_raw_all = explainer.shap_values(X_test_sc)
if isinstance(shap_raw_all, list):
    shap_mat_all = shap_raw_all[1]
elif np.array(shap_raw_all).ndim == 3:
    shap_mat_all = np.array(shap_raw_all)[:, :, 1]
else:
    shap_mat_all = np.array(shap_raw_all)

mean_abs_shap = pd.Series(
    np.abs(shap_mat_all).mean(axis=0), index=X_test_sc.columns
).sort_values(ascending=False)

fig_shap_g, ax_sg = plt.subplots(figsize=(7, 3))
ax_sg.barh(mean_abs_shap.index[::-1], mean_abs_shap.values[::-1],
           color="#1565C0", alpha=0.85)
ax_sg.set_title("Mean |SHAP| Feature Importance (Test Set)", fontsize=12)
ax_sg.set_xlabel("Mean |SHAP| value")
ax_sg.spines["top"].set_visible(False)
ax_sg.spines["right"].set_visible(False)
plt.tight_layout()
st.pyplot(fig_shap_g)
plt.close(fig_shap_g)

st.caption(
    "Model 1: GradientBoostingClassifier (500 trees, lr=0.05, depth=5, "
    "subsample=0.85, sample_weight=balanced)  |  "
    "Model 2: Flan-T5-Small + LoRA/PEFT (3 epochs, 300 SHAP-grounded CoT examples)  |  "
    "Self-Refine: ≤3 rounds (threshold=0.67)  |  "
    "Hard constraint post-correction  |  "
    "Dataset: LSAC Law School Admission (full dataset, stratified split)"
)
