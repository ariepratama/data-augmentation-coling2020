import unittest

from nltk import Tree

from augment_clu_synthetict import find_all_ner_spans, tree_to_synthetic_ner_tree, pre_leaves
from data import Sentence, Token


class TestAugmentCluSyntheticT(unittest.TestCase):
    def test_find_all_ner_spans(self):
        sample_sentence_text = "Within the past year , these symptoms have progressively gotten worse , to encompass also her feet .".split()
        sample_sentence_tags = "O O O O O B-problem I-problem O O O O O O O O O O O".split()
        sample_sentence = Sentence("sample-sentence-1")
        for i, (token_text, label) in enumerate(zip(sample_sentence_text, sample_sentence_tags)):
            token = Token(token_text, i)
            token.set_label("gold", label)
            sample_sentence.add_token(token)
        spans = find_all_ner_spans(sample_sentence)
        self.assertEqual(1, len(spans))
        self.assertEqual(5, spans[0][0])
        self.assertEqual(6, spans[0][1])

    def test_find_all_ner_spans_with_2_tags(self):
        sample_sentence_text = "Within the past year , these symptoms have progressively gotten worse , to encompass also her feet .".split()
        sample_sentence_tags = "O O O O O B-problem I-problem O O O O B-test I-test I-test O O O O".split()
        sample_sentence = Sentence("sample-sentence-1")
        for i, (token_text, label) in enumerate(zip(sample_sentence_text, sample_sentence_tags)):
            token = Token(token_text, i)
            token.set_label("gold", label)
            sample_sentence.add_token(token)
        spans = find_all_ner_spans(sample_sentence)
        self.assertEqual(2, len(spans))
        self.assertEqual(5, spans[0][0])
        self.assertEqual(6, spans[0][1])
        self.assertEqual(11, spans[1][0])
        self.assertEqual(13, spans[1][1])

    def test_find_all_ner_spans_with_tag_in_the_end(self):
        sample_sentence_text = "Within the past year , these symptoms have progressively gotten worse , to encompass also her feet .".split()
        sample_sentence_tags = "O O O O O O O O O O O O O O O B-test I-test I-test".split()
        sample_sentence = Sentence("sample-sentence-1")
        for i, (token_text, label) in enumerate(zip(sample_sentence_text, sample_sentence_tags)):
            token = Token(token_text, i)
            token.set_label("gold", label)
            sample_sentence.add_token(token)
        spans = find_all_ner_spans(sample_sentence)
        self.assertEqual(1, len(spans))
        self.assertEqual(15, spans[0][0])
        self.assertEqual(17, spans[0][1])

    def test_find_all_ner_spans_with_no_tag(self):
        sample_sentence_text = "Within the past year , these symptoms have progressively gotten worse , to encompass also her feet .".split()
        sample_sentence_tags = "O O O O O O O O O O O O O O O O O O".split()
        sample_sentence = Sentence("sample-sentence-1")
        for i, (token_text, label) in enumerate(zip(sample_sentence_text, sample_sentence_tags)):
            token = Token(token_text, i)
            token.set_label("gold", label)
            sample_sentence.add_token(token)
        spans = find_all_ner_spans(sample_sentence)
        self.assertEqual(0, len(spans))

    def test_tree_to_synthetic_ner_tree(self):
        sample_sentence = Sentence("sample_sentence_1")
        sample_sentence_tokens = "doing the thing".split()
        sample_sentence_token_tags = "O B-something I-something".split()
        for idx, (token_text, tag) in enumerate(zip(sample_sentence_tokens, sample_sentence_token_tags)):
            token = Token(token_text, idx)
            token.set_label("gold", tag)
            sample_sentence.add_token(token)

        sample_tree = Tree.fromstring("""
            (S
                (VP doing)
                (NP 
                    (DET the)
                    (N thing)))
        """)
        synthetic_tree, ner_spans = tree_to_synthetic_ner_tree(sample_sentence, sample_tree)
        sample_pre_leaves = pre_leaves(synthetic_tree)
        self.assertEqual("NER_B-something_DET", sample_pre_leaves[1].label())
        self.assertEqual("NER_I-something_N", sample_pre_leaves[2].label())





if __name__ == '__main__':
    unittest.main()
