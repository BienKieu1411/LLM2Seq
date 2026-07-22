import pytest
from genbridge.training import assert_tokenizers_compatible


class StubTokenizer:
    def __init__(self, vocab, bos=1, eos=2, pad=0):
        self._vocab = vocab
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad

    def get_vocab(self):
        return dict(self._vocab)


def test_identical_tokenizers_are_compatible():
    assert_tokenizers_compatible(
        StubTokenizer({"a": 0, "b": 1}),
        StubTokenizer({"a": 0, "b": 1}),
    )


def test_same_size_but_different_token_ids_are_rejected():
    with pytest.raises(ValueError, match="token-to-id vocabularies"):
        assert_tokenizers_compatible(
            StubTokenizer({"a": 0, "b": 1}),
            StubTokenizer({"a": 1, "b": 0}),
        )


def test_different_special_token_roles_are_rejected():
    with pytest.raises(ValueError, match="eos_token_id"):
        assert_tokenizers_compatible(
            StubTokenizer({"a": 0, "b": 1}, eos=1),
            StubTokenizer({"a": 0, "b": 1}, eos=2),
        )
