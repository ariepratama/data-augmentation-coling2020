import logging
import sys

import torch

from augment_bert import augment_sentence_wt_lm
from data import ConllCorpus

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    logging.info("loading corpus...")
    corpus = ConllCorpus("development", "data/M-sent/train.txt", "data/M-sent/dev.txt", "data/M-sent/test.txt")
    tag_dict = corpus.build_tag_dict("gold")
    final_sentences = []
    for sentence in corpus.train:
        logging.info(f"Original sentence={sentence}")
        augmented_sentences, token_idx_candidates, replacement_tokens = augment_sentence_wt_lm(sentence,
                                                                                               n_replacement=5,
                                                                                               num_generated_samples=20)

        for s, t, r in zip(augmented_sentences, token_idx_candidates, replacement_tokens):
            logging.info(f"Generated sentence {s}, {t}, {r}")
        final_sentences += augmented_sentences
    gold_tag_ids = [[tag_dict.get_idx(t.get_label("gold")) for t in s] for s in final_sentences]
    for s in final_sentences:
        for token in s:
            label = token.get_label("gold")
            tag_id = tag_dict.get_idx(token.get_label("gold"))
            if tag_id is None:
                print(f"label = {label}, tag_id = {tag_id}, tag_dict={tag_dict}, sent={s}")
