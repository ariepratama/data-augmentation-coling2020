import unittest

from augment_bert import augment_sentence_wt_lm
from data import Sentence, Token


class TestAugmentBert(unittest.TestCase):
    def test_augment_sentence(self):
        sample_sentence_text = "Within the past year , these symptoms have progressively gotten worse , to encompass also her feet .".split()
        sample_sentence_tags = "O O O O O B-problem I-problem O O O O O O O O O O O".split()
        sample_sentence = Sentence("sample-sentence-1")
        for i, (token_text, label) in enumerate(zip(sample_sentence_text, sample_sentence_tags)):
            token = Token(token_text, i)
            token.set_label("gold", label)
            sample_sentence.add_token(token)

        augmented_sentence, masked_token_id_candidates, mask_replacement_tokens = augment_sentence_wt_lm(
            sample_sentence, seed=928, n_replacement=3)

        print("augmented sentence: ", augmented_sentence)
        print("token id candidates: ", masked_token_id_candidates)
        print("replacement tokens: ", mask_replacement_tokens)


if __name__ == '__main__':
    unittest.main()
