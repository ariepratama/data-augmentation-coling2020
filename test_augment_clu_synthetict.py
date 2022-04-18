import unittest

from augment_clu_synthetict import find_all_ner_spans
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


if __name__ == '__main__':
    unittest.main()
