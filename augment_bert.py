import logging
from typing import *

import numpy as np
import torch
from nltk import word_tokenize
from transformers import BertTokenizer, BertForMaskedLM

from data import Sentence, Token

TOKENIZER = None
MODEL = None


def get_tokenizer(model_name):
    global TOKENIZER

    if TOKENIZER is None:
        TOKENIZER = BertTokenizer.from_pretrained(model_name)
    return TOKENIZER


def get_model(model_name, device):
    global MODEL

    if MODEL is None:
        MODEL = BertForMaskedLM.from_pretrained(model_name).to(device)

    return MODEL


def augment_sentence_wt_lm(sentence: Sentence,
                           model_name="allenai/scibert_scivocab_uncased",
                           seed=33,
                           n_replacement=1,
                           num_generated_samples=1,
                           cuda_device=0) -> Tuple[List[Sentence], List[List[int]], List[List[Text]]]:
    device = torch.device("cuda:%d" % cuda_device)
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, device)

    masked_token_id_candidates = get_token_idx_wt_no_ner(sentence)
    if len(masked_token_id_candidates) == 0:
        logging.warn("will not augment sentence, since no non NER has been replaced")
        return [], [], []

    augmented_sentences = []
    token_idx_candidates = []
    replacement_tokens = []
    np.random.seed(seed)
    for generation_id in range(num_generated_samples):
        masked_token_id_candidates = np.random.choice(masked_token_id_candidates, size=n_replacement)

        tokens = sentence_to_tokens(sentence)
        tokens = mask_token_by_ids(tokens, masked_token_id_candidates)

        inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        mask_replacement_tokens = word_tokenize(tokenizer.decode(predicted_token_id))
        augmented_sentence = sentence_wt_replacement(sentence, masked_token_id_candidates, mask_replacement_tokens,
                                                     generation_id=generation_id)
        logging.info(
            f"Generated sentence: {augmented_sentence}, masked_token_id_candidates: {masked_token_id_candidates}, mask_replacement_tokens: {mask_replacement_tokens}")
        augmented_sentences += [augmented_sentence]
        token_idx_candidates += [masked_token_id_candidates]
        replacement_tokens += [mask_replacement_tokens]
    return augmented_sentences, token_idx_candidates, replacement_tokens


def sentence_wt_replacement(original_sentence: Sentence, replace_ids: List[int], replacement_tokens: List[Text],
                            generation_id=0) -> Sentence:
    id_to_replacement_token = {
        id: replacement
        for id, replacement in zip(replace_ids, replacement_tokens)
    }
    original_sentence_id = original_sentence.idx
    replaced_sentence = Sentence(f"{original_sentence_id}-generated-{generation_id}")
    for token_idx, token in enumerate(original_sentence):
        token_text = token.text
        if token_idx in id_to_replacement_token:
            token_text = id_to_replacement_token[token_idx]
        new_token = Token(token_text, token.idx)
        new_token.set_label("gold", token.get_label("gold"))
        replaced_sentence.add_token(new_token)
    return replaced_sentence


def sentence_to_tokens(sentence: Sentence) -> List[Text]:
    return [tok.text for tok in sentence]


def mask_token_by_ids(tokens: List[Text], mask_ids: List[int], mask_token="[MASK]") -> List[Text]:
    for mask_id in mask_ids:
        tokens[mask_id] = mask_token
    return tokens


def get_token_idx_wt_no_ner(sentence: Sentence) -> List[int]:
    return [
        token_idx for token_idx, token in enumerate(sentence)
        if token.get_label("gold") == "O"
    ]
