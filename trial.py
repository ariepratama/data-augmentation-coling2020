import logging
import sys
import torch

from data import ConllCorpus
from augment_clu_synthetict import generate_sentences_by_synthetic_tree

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    logging.info("loading corpus...")
    corpus = ConllCorpus("development", "data/S-sent/train.txt", "data/S-sent/dev.txt", "data/S-sent/test.txt")
    tag_dict = corpus.build_tag_dict("gold")
    final_sentences = []
    for sentence in corpus.train:
        logging.info(f"Original sentence={sentence}")
        augmented_sentences = generate_sentences_by_synthetic_tree(sentence, 10, corpus.train, "NP",
                                                                   n_replaced_non_terminal=5, random_state=300)
        final_sentences += augmented_sentences
    gold_tag_ids = [[tag_dict.get_idx(t.get_label("gold")) for t in s] for s in final_sentences]
    for s in final_sentences:
        for token in s:
            label = token.get_label("gold")
            tag_id = tag_dict.get_idx(token.get_label("gold"))
            if tag_id is None:
                print(f"label = {label}, tag_id = {tag_id}, tag_dict={tag_dict}, sent={s}")
