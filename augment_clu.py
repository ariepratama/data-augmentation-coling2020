import logging
from typing import *

from clu.daug.chunker import Chunker
from clu.daug.mutator import RandomSelectedTreesXPMutator
from clu.daug.selector import RandomWalkSelector
from nltk import Tree

from data import Sentence, Token


def corpus_to_trees(dataset) -> List[Tree]:
    trees: List[Tree] = []
    chunker: Chunker = Chunker.default()

    for sentence in dataset:
        token_texts = [token.text for token in sentence]
        sentence_tree = chunker.transform(token_texts)[0]
        trees.append(sentence_tree)

    return trees


def generate_sentences_by_grammar(sentence: Sentence,
                                  num_generated_samples: int,
                                  non_terminal: str,
                                  corpus_tree: List[Tree],
                                  idx2sentence: Dict[str, Sentence],
                                  random_state: int = None,
                                  is_dev_mode: bool = False) -> List[Sentence]:
    """
    for category2mentions: see get_category2mentions
    sample
    {
        'problem': ['C5-6 disc herniation'],
        'test': ['outpatient Holter monitor']
    }
    """
    chunker: Chunker = Chunker.default(non_terminal=non_terminal, is_dev_mode=is_dev_mode)

    sentences: List[Sentence] = []
    token_texts = [token.text for token in sentence.tokens]
    sentence_tree = chunker.transform(token_texts)[0]

    actual_n_generated_samples = min(len(token_texts), num_generated_samples)
    selector = RandomWalkSelector(actual_n_generated_samples)
    mutator = RandomSelectedTreesXPMutator(non_terminal, corpus_tree)
    logging.info(f"Original sentence: {sentence}")
    for generation_idx, (target_parent, target_child_idx) in enumerate(
            selector.select(sentence_tree, actual_n_generated_samples)):
        logging.info(f"selected sentence from selector, target_parent={target_parent} target_child_idx={target_child_idx}")
        mutated_parent, _, selected_corpus_indexes = mutator.transform(target_parent, target_child_idx,
                                                                       random_seed=random_state)
        tok2tag = generate_tok2label(idx2sentence, selected_corpus_indexes, sentence)
        # get start and end of original sentence's mutated span
        start_mutated_idx, end_mutated_idx = get_start_end_of_span(target_parent, target_child_idx)
        generated_sentence = to_sentence(sentence, generation_idx, mutated_parent.leaves(),
                                         start_mutated_idx, end_mutated_idx, tok2tag)

        logging.info(f"Generated sentence-{generation_idx}: {generated_sentence}")
        sentences.append(generated_sentence)
    return sentences


def get_start_end_of_span(root, selected_parent_index) -> Tuple[int, int]:
    target_node = root[selected_parent_index]
    leaves = []
    return get_start_end_of_span_rec(root, target_node, leaves)


def get_start_end_of_span_rec(explored_node, target_node, leaves):
    # base case
    if explored_node == target_node:
        selected_start_span_idx = len(leaves)
        if type(explored_node) == Tree:
            selected_end_span_idx = selected_start_span_idx + max(len(explored_node.leaves()) - 1, 0)
            return selected_start_span_idx, selected_end_span_idx

    if type(explored_node) == Tree:
        # doing dfs
        for subtree_idx, subtree in enumerate(explored_node):
            selected_start_span_idx, selected_end_span_idx = get_start_end_of_span_rec(subtree, target_node, leaves)
            if selected_start_span_idx != -1 and selected_end_span_idx != -1:
                return selected_start_span_idx, selected_end_span_idx

        return -1, -1
    leaves.append(explored_node)

    return -1, -1


def generate_tok2label(idx2sentence, selected_corpus_indexes, original_sentence):
    tok2label = {}
    selected_corpus_sentences = [idx2sentence[corpus_idx] for corpus_idx in selected_corpus_indexes]
    selected_corpus_sentences += [original_sentence]
    for corpus_sentence in selected_corpus_sentences:
        for token in corpus_sentence:
            if token.text not in tok2label:
                label = token.get_label("gold")
                tok2label[token.text] = label
                if "-" in token.text:
                    splitted_tokens = token.text.split("-")
                    splitted_labels = [label] * len(splitted_tokens)
                    if label.startswith("B"):
                        category = label.split("-")[-1]
                        for i in range(1, len(splitted_labels)):
                            splitted_labels[i] = f"I-{category}"

                    for new_token, new_label in zip(splitted_tokens, splitted_labels):
                        tok2label[new_token] = new_label
            else:
                logging.debug(f"`{token.text}` is already on tok2label, skipping...")
    return tok2label


def to_sentence(original_sentence, n_generated_sentence, leaves,
                start_mutated_idx, end_mutated_idx, tok2tag) -> Sentence:
    sent = Sentence(f"{original_sentence.idx}-replace-by-grammar-{n_generated_sentence}")
    bm_start, bm_end, am_start, am_end = find_before_and_after_mutation_idx(original_sentence, leaves,
                                                                            start_mutated_idx, end_mutated_idx)
    sentence_length_diff = len(leaves) - len(original_sentence.tokens)
    logging.info(f"original sentence = {original_sentence}")
    logging.info(f"mutation index: start={start_mutated_idx}, end={end_mutated_idx}")
    logging.info(f"mutated leaves = {leaves}")

    for token_idx, leave in enumerate(leaves):
        text = leave
        if type(leave) == tuple:
            text = leave[0]
        label = "O"
        if start_mutated_idx <= token_idx <= end_mutated_idx + sentence_length_diff:
            label = tok2tag[text]
        elif -1 < bm_start <= token_idx < bm_end + 1:
            label = original_sentence.get_token(token_idx).get_label("gold")
        elif -1 < am_start <= token_idx:
            new_token_idx = token_idx - sentence_length_diff
            logging.info(f"new_token_idx={new_token_idx}, am_start={am_start}, token_idx={token_idx}, end_mutated_idx={end_mutated_idx}, s={sentence_length_diff}")
            label = original_sentence.get_token(new_token_idx).get_label("gold")

        token = Token(text)
        token.set_label("gold", label)

        sent.add_token(token)

    return sent


def find_before_and_after_mutation_idx(original_sentence, leaves, start_mutated_idx, end_mutated_idx) -> Tuple[
    int, int, int, int]:
    # mutation happened at the beginning of sentence
    if start_mutated_idx == 0:
        return -1, -1, end_mutated_idx + 1, len(leaves) - 1

    # mutation happened from middle to the end of sentence
    if end_mutated_idx >= len(leaves):
        return 0, start_mutated_idx - 1, -1, -1

    bm_start, bm_end = 0, 0
    bm_end = start_mutated_idx - 1
    logging.info(f"leaves: {leaves}, start: {start_mutated_idx} end: {end_mutated_idx}")
    first_after_mutation_token_text = leaves[end_mutated_idx]

    first_after_mutation_token_original_idx = 0
    for original_token_idx, token in enumerate(original_sentence[start_mutated_idx:]):
        if token.text == first_after_mutation_token_text:
            first_after_mutation_token_original_idx = original_token_idx + start_mutated_idx
            break

    return bm_start, bm_end, first_after_mutation_token_original_idx, -1
