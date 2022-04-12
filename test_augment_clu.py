import logging
import sys
import unittest
from collections import defaultdict

from nltk import Tree

from augment_clu import get_start_end_of_span, find_before_and_after_mutation_idx, to_sentence
from data import Sentence, Token

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestAugmentClu(unittest.TestCase):
    # def test_generate_sentences_by_grammar(self):
    #     idx = "some-idx-1"
    #     sentence = Sentence(idx)
    #     for token_text in "The patient was given printed instruction for dizziness.".split():
    #         sentence.add_token(Token(token_text))
    #     corpus_trees = [
    #         Tree.fromstring("""
    #                             (S
    #                                 (NP (PRP I))
    #                                 (VP
    #                                     (VBP am)
    #                                     (VP
    #                                         (VP
    #                                             (VBG eating))
    #                                             (INTJ (SYM sandwich)))))
    #                             """),
    #         Tree.fromstring("""
    #                             (INTJ (INTJ (VB please)) (VP (VB submit) (NP (DT some) (NNS homeworks)) (PP (IN by) (NP (CD 11) (<STOP> am)))))
    #                             """)
    #     ]
    #     sentences = generate_sentences_by_grammar(sentence, 1, "NP", corpus_trees, {}, is_dev_mode=True)
    #
    #     self.assertTrue(sentences is not None)
    #     print(sentences)

    def test_get_start_end_of_span(self):
        sample_tree = Tree.fromstring("""
                                (S 
                                    (NP (PRP I)) 
                                    (VP 
                                        (VBP am) 
                                        (VP 
                                            (VP 
                                                (VBG eating)) 
                                                (INTJ (SYM sandwich))))
                                    (TO to)
                                    (VP (VB fill) (PRP me) (PRP up))
                                )
                                """)

        start, end = get_start_end_of_span(sample_tree, 2)
        leaves = sample_tree.leaves()
        self.assertEqual(leaves[start:end + 1], ["to"])

        start, end = get_start_end_of_span(sample_tree, 1)
        self.assertEqual(leaves[start:end + 1], ["am", "eating", "sandwich"])

        start, end = get_start_end_of_span(sample_tree, 3)
        self.assertEqual(leaves[start:end + 1], ["fill", "me", "up"])

    def test_find_before_and_after_mutation_idx_original_sentence_is_longer(self):
        sample_sentence_text = "this sentence is ready to be mutated but this part has not"
        start_mutated_idx = 2
        end_mutated_idx = 5
        sample_sentence = Sentence("sample-sentence-1")
        for i, token_text in enumerate(sample_sentence_text.split()):
            token = Token(token_text, i)
            sample_sentence.add_token(token)
        leaves = "this sentence has been mutated but this part has not".split()

        bm_start, bm_end, am_start, am_end = find_before_and_after_mutation_idx(sample_sentence, leaves,
                                                                                start_mutated_idx, end_mutated_idx)
        print(bm_start, bm_end, am_start, am_end)
        self.assertEqual(bm_start, 0)
        self.assertEqual(bm_end + 1, start_mutated_idx)
        self.assertNotEqual(am_start, end_mutated_idx)
        self.assertEqual(am_start, 7)

    def test_find_before_and_after_mutation_idx_original_sentence_is_shorter(self):
        sample_sentence_text = "this sentence is mutating but this part has not"
        start_mutated_idx = 2
        end_mutated_idx = 5
        sample_sentence = Sentence("sample-sentence-1")
        for i, token_text in enumerate(sample_sentence_text.split()):
            token = Token(token_text, i)
            sample_sentence.add_token(token)
        leaves = "this sentence has been mutated but this part has not".split()

        bm_start, bm_end, am_start, am_end = find_before_and_after_mutation_idx(sample_sentence, leaves,
                                                                                start_mutated_idx, end_mutated_idx)
        print(bm_start, bm_end, am_start, am_end)
        self.assertEqual(bm_start, 0)
        self.assertEqual(bm_end + 1, start_mutated_idx)
        self.assertNotEqual(am_start, end_mutated_idx)
        self.assertEqual(am_start, 4)

    def test_to_sentence(self):
        sample_sentence_text = "this sentence is mutating but this part has not"
        sample_sentence = Sentence("sample-sentence-1")
        for i, token_text in enumerate(sample_sentence_text.split()):
            token = Token(token_text, i)
            token.set_label("gold", "O")
            sample_sentence.add_token(token)

        start_mutated_idx = 2
        end_mutated_idx = 4
        leaves = "this sentence has been mutated but this part has not".split()
        tok2tag = defaultdict(str)
        tok2tag["has"] = "B"
        tok2tag["been"] = "B"
        tok2tag["mutated"] = "B"

        new_sentence = to_sentence(sample_sentence, 1, leaves, start_mutated_idx, end_mutated_idx, tok2tag)

        print(new_sentence)
        for token in new_sentence[:start_mutated_idx]:
            self.assertEqual(token.get_label("gold"), "O")

        for token in new_sentence[start_mutated_idx:end_mutated_idx + 1]:
            self.assertEqual(token.get_label("gold"), "B")

        for token in new_sentence[end_mutated_idx + 1:]:
            self.assertEqual(token.get_label("gold"), "O")

    def test_to_sentence_with_mutated_at_similar_length(self):
        sample_sentence_text = "this sentence is mutating but this part has not"
        sample_sentence = Sentence("sample-sentence-1")
        for i, token_text in enumerate(sample_sentence_text.split()):
            token = Token(token_text, i)
            token.set_label("gold", "O")
            sample_sentence.add_token(token)

        start_mutated_idx = 2
        end_mutated_idx = 3
        leaves = "this sentence did mutated but this part has not".split()
        tok2tag = defaultdict(str)
        tok2tag["did"] = "B"
        tok2tag["mutated"] = "B"

        new_sentence = to_sentence(sample_sentence, 1, leaves, start_mutated_idx, end_mutated_idx, tok2tag)
        print(new_sentence)

        for t in new_sentence:
            print(t.text, t.get_label("gold"))

        for token in new_sentence[:start_mutated_idx]:
            self.assertEqual(token.get_label("gold"), "O")

        for token in new_sentence[start_mutated_idx:end_mutated_idx + 1]:
            self.assertEqual(token.get_label("gold"), "B")

        for token in new_sentence[end_mutated_idx + 1:]:
            self.assertEqual(token.get_label("gold"), "O")

    def test_to_sentence_with_mutated_at_similar_length_and_leaves_is_longer(self):
        sample_sentence_text = "this sentence is mutating but this part has not"
        sample_sentence = Sentence("sample-sentence-1")
        for i, token_text in enumerate(sample_sentence_text.split()):
            token = Token(token_text, i)
            token.set_label("gold", "O")
            sample_sentence.add_token(token)

        start_mutated_idx = 7
        end_mutated_idx = 11
        leaves = "this sentence is mutating but this part is now has been mutated".split()
        tok2tag = defaultdict(str)
        tok2tag["is"] = "B"
        tok2tag["now"] = "B"
        tok2tag["has"] = "B"
        tok2tag["been"] = "B"
        tok2tag["mutated"] = "B"

        new_sentence = to_sentence(sample_sentence, 1, leaves, start_mutated_idx, end_mutated_idx, tok2tag)

        for t in new_sentence:
            print(t.text, t.get_label("gold"))

        for token in new_sentence[:start_mutated_idx]:
            self.assertEqual(token.get_label("gold"), "O")

        for token in new_sentence[start_mutated_idx:end_mutated_idx + 1]:
            self.assertEqual(token.get_label("gold"), "B")

    def test_to_sentence_with_mutated_at_beginning_and_leaves_is_longer(self):
        sample_sentence_text = "this sentence is mutating but this part has not"
        sample_sentence = Sentence("sample-sentence-1")
        for i, token_text in enumerate(sample_sentence_text.split()):
            token = Token(token_text, i)
            token.set_label("gold", "O")
            sample_sentence.add_token(token)

        start_mutated_idx = 0
        end_mutated_idx = 3
        leaves = "that previous sentence a is mutating but this part has not".split()
        tok2tag = defaultdict(str)
        tok2tag["that"] = "B"
        tok2tag["previous"] = "B"
        tok2tag["sentence"] = "B"
        tok2tag["a"] = "B"

        new_sentence = to_sentence(sample_sentence, 1, leaves, start_mutated_idx, end_mutated_idx, tok2tag)

        for t in new_sentence:
            print(t.text, t.get_label("gold"))

        for token in new_sentence[:start_mutated_idx]:
            self.assertEqual(token.get_label("gold"), "O")

        for token in new_sentence[start_mutated_idx:end_mutated_idx + 1]:
            self.assertEqual(token.get_label("gold"), "B")


if __name__ == '__main__':
    unittest.main()
