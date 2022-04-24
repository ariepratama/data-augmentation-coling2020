import unittest

from nltk import Tree

from augment_clu_synthetict import find_all_ner_spans, tree_to_synthetic_ner_tree, pre_leaves, find_related_ner_spans
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
                    the/DET
                    thing/N))
        """)
        synthetic_tree, ner_spans = tree_to_synthetic_ner_tree(sample_sentence, sample_tree)

        sample_pre_leaves = pre_leaves(synthetic_tree)
        print(synthetic_tree)
        self.assertEqual("NER_B-something_PLACEHOLDER", sample_pre_leaves[1].label())
        self.assertEqual("NER_I-something_PLACEHOLDER", sample_pre_leaves[2].label())

    def test_find_related_ner_spans(self):
        ner_category = "problem"
        sample_sentence = Sentence("sample_sentence_1")
        sample_sentence_tokens = "doing the thing".split()
        sample_sentence_token_tags = "O B-problem I-problem".split()

        for idx, (token_text, tag) in enumerate(zip(sample_sentence_tokens, sample_sentence_token_tags)):
            token = Token(token_text, idx)
            token.set_label("gold", tag)
            sample_sentence.add_token(token)

        replacement_sentence = Sentence("sample")
        replacement_sentence_tokens = "This love has taken its toll on me , she said .".split()
        replacement_sentence_tags = "O O B-problem I-problem I-problem I-problem O O O O O O".split()

        for idx, (token_text, tag) in enumerate(zip(replacement_sentence_tokens, replacement_sentence_tags)):
            token = Token(token_text, idx)
            token.set_label("gold", tag)
            replacement_sentence.add_token(token)
        replacement_ner_spans = [(2, 5)]
        related_spans = find_related_ner_spans(ner_category, replacement_sentence, replacement_ner_spans)
        self.assertEqual(1, len(related_spans))
        self.assertEqual(related_spans[0], replacement_ner_spans[0])

    def test(self):
        sample_sentence_unparsed = """
            Do O
            not O
            drive O
            if O
            taking O
            flexeril B-treatment
            and O
            codeine B-treatment
            . O
            NEAR B-problem
            SYNCOPE I-problem
            Standardized O
            Discharge O
            Instructions O
            : O
        """
        sample_tree = Tree.fromstring("""
        (S
          Do/VB
          not/RB
          (NP drive/VB if/IN)
          (NP taking/VBG flexeril/NN)
          (NP and/CC codeine/NN)
          ./.
          NEAR/NNP
          (NP SYNCOPE/NNP Standardized/JJ)
          (NP Discharge/NNP Instructions/NNPS :/:))
        """)
        sample_sentence = Sentence("sample-sentence")
        for line in sample_sentence_unparsed.split("\n"):
            line = line.strip()
            if len(line) == 0:
                continue
            token_text, tag = line.split()
            token = Token(token_text)
            token.set_label("gold", tag)
            sample_sentence.add_token(token)

        synthetic_tree, ner_spans = tree_to_synthetic_ner_tree(sample_sentence, sample_tree)
        self.assertEqual(3, len(ner_spans))
        self.assertEqual("NER_B-treatment_PLACEHOLDER", synthetic_tree[3][1].label())
        self.assertEqual("NER_B-treatment_PLACEHOLDER", synthetic_tree[4][1].label())
        self.assertEqual("NER_B-problem_PLACEHOLDER", synthetic_tree[6].label())


if __name__ == '__main__':
    unittest.main()
