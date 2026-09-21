"""
Model utilities for loading BERT and extracting predictions + attention.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True
    )
    model.eval()
    return tokenizer, model


def predict_proba(texts, tokenizer, model):
    """Return class probabilities for a list of texts. Required by LIME."""
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs.numpy()


def get_attention_and_tokens(text, tokenizer, model):
    """
    Run a single text through the model and return:
      - tokens (subword tokens, including special tokens)
      - a per-token importance score derived from attention

    Aggregation choice: mean attention paid FROM the [CLS] token, averaged
    across all heads of the LAST layer. [CLS] is what feeds the
    classification head, so this approximates "what the model looked at
    when it made its decision."
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    last_layer_attn = outputs.attentions[-1][0]        # (num_heads, seq, seq)
    cls_attn = last_layer_attn[:, 0, :]                 # attention FROM [CLS]
    token_importance = cls_attn.mean(dim=0).numpy()     # mean over heads

    return tokens, token_importance
