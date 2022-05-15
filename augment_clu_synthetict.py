import logging
import random
from typing import *

from clu.daug.chunker import Chunker
from nltk import Tree, ParentedTree

from data import Sentence, Token

SYNTHETIC_TREES_CACHE = {}
NER_SPANS_CACHE = {}
CATEGORY_TO_SENTENCE_ID_MAP = {}
SENTENCE_ID_TO_SENTENCE = {}
SENTENCE_ID_TO_NER_NONTERMINAL_NODES = {}


def corpus_to_trees(dataset) -> List[Tree]:
    trees: List[Tree] = []
    chunker: Chunker = Chunker.default()

    for sentence in dataset:
        token_texts = [token.text for token in sentence]
        sentence_tree = chunker.transform(token_texts)[0]
        trees.append(sentence_tree)

    return trees


def generate_sentences_by_synthetic_tree(sentence: Sentence,
                                         num_generated_samples: int,
                                         train_corpus: List[Sentence],
                                         non_terminal: str,
                                         n_replaced_non_terminal: int = 1,
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
    generated_sentences = []
    if len(SYNTHETIC_TREES_CACHE.keys()) <= 0:
        populate_caches(sentence, train_corpus, non_terminal, is_dev_mode)

    if sentence.idx not in SYNTHETIC_TREES_CACHE:
        logging.info(f"aborting augmentation because cache the sentence {sentence} is not found on cache...")
        return [sentence]

    original_sentence_id = sentence.idx
    sentence_tree = SYNTHETIC_TREES_CACHE[original_sentence_id]
    ner_spans = NER_SPANS_CACHE[original_sentence_id]
    original_sentence_nonterminal_nodes = SENTENCE_ID_TO_NER_NONTERMINAL_NODES[original_sentence_id]

    logging.info(f"Original sentence: \"{sentence}\", original sentence tree: {sentence_tree}")
    logging.info(f"Original non-terminal nodes: {original_sentence_nonterminal_nodes}")
    if num_generated_samples == 0 or len(ner_spans) == 0 or len(original_sentence_nonterminal_nodes) == 0:
        logging.info(
            f"Will not generate more sentences because num_generated_samples={num_generated_samples}" +
            f" ner_spans={ner_spans}")
        return generated_sentences

    if random_state:
        random.seed(random_state)

    # random sample with replacement
    nodes_to_be_replaced: List[ParentedTree] = random.choices(
        original_sentence_nonterminal_nodes,
        k=n_replaced_non_terminal
    )

    for generation_id in range(num_generated_samples):
        # need to refresh sentence_tree before generating new sentence

        mutated_sentence_tree = SYNTHETIC_TREES_CACHE[original_sentence_id].copy(deep=True)
        for node_to_be_replaced in nodes_to_be_replaced:
            replacement_sentence_id_candidates = set(SENTENCE_ID_TO_SENTENCE.keys())
            replacement_sentence_id_candidates.remove(original_sentence_id)
            replacement_sentence_id = random.choice(list(replacement_sentence_id_candidates))
            related_nodes_from_replacement_sentence = SENTENCE_ID_TO_NER_NONTERMINAL_NODES[replacement_sentence_id]
            if len(related_nodes_from_replacement_sentence) < 1:
                logging.warning("cannot find related nodes to replace...")
                continue
            replacement_node = random.choice(related_nodes_from_replacement_sentence)
            mutated_sentence_tree[node_to_be_replaced.treeposition()] = ParentedTree.convert(
                replacement_node.copy(deep=True))
        generated_sentence = tree_to_sentence(mutated_sentence_tree, original_sentence_id, generation_id)

        if generated_sentence is None:
            logging.warn("Will not generate sentence, cannot find related_ner_spans")
            continue

        logging.info(f"Generated sentence: \"{generated_sentence}\", generated tree: {mutated_sentence_tree}")
        generated_sentences.append(generated_sentence)

    return generated_sentences


def tree_to_sentence(tree: Tree, original_sentence_id: str, generation_id: int) -> Sentence:
    parented_tree = ParentedTree.convert(tree)
    tree_pre_leaves = pre_leaves(parented_tree)
    sentence = Sentence(f"{original_sentence_id}-generated-{generation_id}")

    for i, pre_leaf in enumerate(tree_pre_leaves):
        token_text = pre_leaf[
            parented_tree.leaf_treeposition(i)[-1]
        ]
        if type(token_text) == tuple:
            token_text = token_text[0]
        try:
            token_text = token_text.split("/")[0]
        except Exception as e:
            logging.error(f"something happens here, token_text={token_text}", e)
        token_label = "O"
        if "NER" in pre_leaf.label():
            token_label = pre_leaf.label().split("_")[1]
        token = Token(token_text, i)
        token.set_label("gold", token_label)
        sentence.add_token(token)

    return sentence


def populate_caches(sentence, train_corpus, non_terminal, is_dev_mode):
    # logging.info("populating SENTENCE_ID_TO_SENTENCE cache...")
    for train_sentence in train_corpus:
        SENTENCE_ID_TO_SENTENCE[train_sentence.idx] = train_sentence

    data = corpus_to_synthetic_trees(train_corpus, non_terminal=non_terminal, is_dev_mode=is_dev_mode)
    for sentence_id, (synthetic_tree, ner_spans, train_sentence) in data:
        SYNTHETIC_TREES_CACHE[sentence_id] = synthetic_tree
        SENTENCE_ID_TO_NER_NONTERMINAL_NODES[sentence_id] = find_related_ner_nonterminal_nodes(
            synthetic_tree,
            non_terminal
        )
        NER_SPANS_CACHE[sentence_id] = ner_spans

        for start_ner_span, _ in ner_spans:
            start_ner_token = train_sentence[start_ner_span]
            category = start_ner_token.get_label("gold").split("-")[-1]

            if category not in CATEGORY_TO_SENTENCE_ID_MAP:
                CATEGORY_TO_SENTENCE_ID_MAP[category] = []

            CATEGORY_TO_SENTENCE_ID_MAP[category].append(sentence.idx)


def find_related_ner_nonterminal_nodes(synthetic_tree: Tree, non_terminal: str) -> List[Tree]:
    """
    Search all nodes that have this format: NER_{non_terminal} from a parented tree (synthetic tree).
    """
    if not isinstance(synthetic_tree, ParentedTree):
        synthetic_tree = ParentedTree.convert(synthetic_tree)

    visited_set = set()
    exploration_queue = [synthetic_tree]
    result = []
    while not len(exploration_queue) == 0:
        node_to_explore = exploration_queue.pop()
        visited_set.add(node_to_explore.treeposition())

        if node_to_explore.label() == f"NER_{non_terminal}":
            result.append(node_to_explore)

        for child in node_to_explore:
            if isinstance(child, Tree) and child.treeposition() not in visited_set:
                exploration_queue.append(child)

    return result


def find_ner_node_given_span(tree: Tree, start_span: int, non_terminal: str) -> ParentedTree:
    parented_tree = tree
    # do not convert if the tree has already had parents
    if not isinstance(tree, ParentedTree):
        parented_tree = ParentedTree.convert(tree.copy())

    tree_pre_leaves = pre_leaves(parented_tree)
    node_candidate = tree_pre_leaves[start_span]

    while "NER" not in node_candidate.label() and node_candidate.parent() is not None:
        node_candidate = node_candidate.parent()

    return node_candidate


def find_related_ner_spans(ner_category: Text, replacement_sentence: Sentence,
                           replacement_ner_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    results = []
    for start_ner_span, end_ner_span in replacement_ner_spans:
        replacement_ner_category = replacement_sentence[start_ner_span].get_label("gold").split("-")[-1]
        if ner_category == replacement_ner_category:
            results.append((start_ner_span, end_ner_span))
    return results


def corpus_to_synthetic_trees(train_corpus: List[Sentence], non_terminal: str, is_dev_mode: bool = False) -> List[
    Tuple[str, Tuple[Tree, List[Tuple[int, int]], Sentence]]]:
    chunker: Chunker = Chunker.default(non_terminal=non_terminal, is_dev_mode=is_dev_mode)

    results: List[str, Tuple[Tree, List[Tuple[int, int]]]] = []
    for sentence in train_corpus:
        token_texts = [token.text for token in sentence.tokens]
        sentence_tree = chunker.transform(token_texts)[0]
        sentence_tree, ner_spans = tree_to_synthetic_ner_tree(sentence, sentence_tree)
        results.append((sentence.idx, (sentence_tree, ner_spans, sentence)))

    return results


def tree_to_synthetic_ner_tree(original_sentence: Sentence, original_sentence_tree: Tree) -> Tuple[
    Tree, List[Tuple[int, int]]]:
    """Add NER information to the tree generated by chunker
    """
    ori_parented_tree = ParentedTree.convert(original_sentence_tree.copy())
    ner_spans = find_all_ner_spans(original_sentence)
    original_pre_leafs = pre_leaves(ori_parented_tree)
    leaf_idx_to_parent_idx = {}

    if len(ner_spans) <= 0:
        return Tree.convert(ori_parented_tree), []

    for i, pre_leaf in enumerate(original_pre_leafs):
        leaf_idx_to_parent_idx[i] = ori_parented_tree.leaf_treeposition(i)[-1]

    for start_span, end_span in ner_spans:
        for i in range(start_span, end_span + 1):
            current_token = original_sentence.get_token(i)
            current_token_gold_label = current_token.get_label("gold")
            # use ParentedTree.treeposition()
            child_idx = leaf_idx_to_parent_idx[i]
            current_leaf = original_pre_leafs[i][child_idx]
            modified_pre_leaf_label = f"NER_{current_token_gold_label}_PLACEHOLDER"
            # will form e.g: (S (NP (NER_B-test_PLACEHOLDER token1/NN) ))
            new_pre_leaf = ParentedTree.convert(Tree(modified_pre_leaf_label, [current_leaf]))
            # replace one
            original_pre_leafs[i][child_idx] = new_pre_leaf

            pre_leaf_parent = original_pre_leafs[i][child_idx]

            # skip the root non-terminal
            while pre_leaf_parent.parent() is not None and pre_leaf_parent.parent().parent() is not None:
                pre_leaf_parent = pre_leaf_parent.parent()
                if pre_leaf_parent.label() is not None and "NER" not in pre_leaf_parent.label():
                    # mark the parents as NER_
                    new_label = f"NER_{pre_leaf_parent.label()}"
                    pre_leaf_parent.set_label(new_label)

    return Tree.convert(ori_parented_tree), ner_spans


def pre_leaves(tree) -> List[Tree]:
    """Because leaves() will return non tree and we cannot search bottoms up, we will need to retrieve all nodes before leaves.
    these pre_leaves will have len equal to leaves.
    """
    result = []
    for child in tree:
        if isinstance(child, Tree) and len(child) > 0:
            result.extend(pre_leaves(child))
        elif not isinstance(child, Tree):
            result.append(tree)

    return result


def find_all_ner_spans(original_sentence: Sentence) -> List[Tuple[int, int]]:
    result = []
    start_idx = -1
    len_token = len(original_sentence)
    for i in range(len_token):
        token = original_sentence[i]
        next_token = None

        if i < len_token - 1:
            next_token = original_sentence[i + 1]

        if token.get_label("gold").startswith("B-"):
            start_idx = i
            continue

        if token.get_label("gold").startswith("I-") and (next_token is None or next_token.get_label("gold") == "O"):
            result.append((start_idx, i))
            start_idx = -1
            continue

        if token.get_label("gold") == "O" and start_idx > 0:
            result.append((start_idx, start_idx))
            start_idx = -1

    return result
