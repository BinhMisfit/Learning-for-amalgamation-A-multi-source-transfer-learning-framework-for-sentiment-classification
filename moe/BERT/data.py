import torch
import torch.utils as utils


def rightpad(array, max_len=128):
    if len(array) > max_len:
        padded_array = array[: max_len]
    else:
        padded_array = array + ([0] * (max_len - len(array)))

    return padded_array


def segment(text, segmenter):
    words = []
    for word in segmenter.tokenize(text):
        words = words + word
    text = ' '.join([word for word in words])

    return text


def encode(text, segmenter, vocab):
    text = segment(text, segmenter)
    text = '<s> ' + text + ' </s>'
    text_ids = vocab.encode_line(text, append_eos=False, add_if_not_exist=False).long().tolist()

    return text_ids


class ReviewDataset(utils.data.Dataset):
    def __init__(self, reviews, ratings, segmenter, vocab, tokenizer):
        self.vocab     = vocab
        self.segmenter = segmenter
        self.tokenizer = tokenizer
        self.dataset   = [
            (
                rightpad(encode(reviews[i], self.segmenter, self.vocab)),
                rightpad(self.tokenizer.encode("[CLS] " + reviews[i] + " [SEP]")),
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_phobert, review_bert, rating = self.dataset[index]
        review_phobert, review_bert, rating = torch.tensor(review_phobert), torch.tensor(review_bert), torch.tensor(rating)


        return review_phobert, review_bert, rating
