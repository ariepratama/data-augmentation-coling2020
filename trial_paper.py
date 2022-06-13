import logging
import sys
import torch

from augment import generate_sentences_by_replace_mention, get_category2mentions, \
    generate_sentences_by_synonym_replacement
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
        category2mentions = get_category2mentions(corpus.train)
        # augmented_sentences = generate_sentences_by_replace_mention(sentence, category2mentions, 0.3,
        #                                                              1)
        augmented_sentences = generate_sentences_by_synonym_replacement(sentence, 0.3, 1)
        final_sentences += augmented_sentences
    gold_tag_ids = [[tag_dict.get_idx(t.get_label("gold")) for t in s] for s in final_sentences]
    for s in final_sentences:
        for token in s:
            label = token.get_label("gold")
            tag_id = tag_dict.get_idx(token.get_label("gold"))
            if tag_id is None:
                print(f"label = {label}, tag_id = {tag_id}, tag_dict={tag_dict}, sent={s}")
