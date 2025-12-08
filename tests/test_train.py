"""
Tests for TinyLLM training components.
TinyLLM学習コンポーネントのテスト。
"""

import pytest
import torch
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import SimpleTokenizer, TextDataset, train_epoch, evaluate
from tiny_transformer import TinyLLM


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer class."""

    def test_init(self):
        """Test SimpleTokenizer initialization."""
        tokenizer = SimpleTokenizer()

        assert tokenizer.vocab_size == 0
        assert tokenizer.char_to_id == {}
        assert tokenizer.id_to_char == {}
        assert tokenizer.pad_token == "<PAD>"
        assert tokenizer.unk_token == "<UNK>"
        assert tokenizer.bos_token == "<BOS>"
        assert tokenizer.eos_token == "<EOS>"

    def test_build_vocab(self):
        """Test vocabulary building."""
        tokenizer = SimpleTokenizer()
        texts = ["hello", "world"]
        tokenizer.build_vocab(texts)

        assert tokenizer.vocab_size > 0
        assert "<PAD>" in tokenizer.char_to_id
        assert "<UNK>" in tokenizer.char_to_id
        assert "<BOS>" in tokenizer.char_to_id
        assert "<EOS>" in tokenizer.char_to_id
        assert "h" in tokenizer.char_to_id
        assert "e" in tokenizer.char_to_id
        assert "l" in tokenizer.char_to_id
        assert "o" in tokenizer.char_to_id
        assert "w" in tokenizer.char_to_id
        assert "r" in tokenizer.char_to_id
        assert "d" in tokenizer.char_to_id

    def test_encode(self):
        """Test text encoding."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        encoded = tokenizer.encode("hello")

        assert len(encoded) == 5
        assert all(isinstance(id, int) for id in encoded)

    def test_encode_unknown_char(self):
        """Test encoding with unknown character."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        encoded = tokenizer.encode("xyz")

        unk_id = tokenizer.char_to_id["<UNK>"]
        assert all(id == unk_id for id in encoded)

    def test_decode(self):
        """Test text decoding."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)

        assert decoded == "hello"

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        tokenizer = SimpleTokenizer()
        text = "The quick brown fox"
        tokenizer.build_vocab([text])

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_special_token_ids(self):
        """Test special token ID properties."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        assert tokenizer.pad_id == tokenizer.char_to_id["<PAD>"]
        assert tokenizer.bos_id == tokenizer.char_to_id["<BOS>"]
        assert tokenizer.eos_id == tokenizer.char_to_id["<EOS>"]

    def test_decode_skips_special_tokens(self):
        """Test that decode skips PAD, BOS, EOS tokens."""
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["hello"])

        ids_with_special = [tokenizer.pad_id, tokenizer.bos_id] + tokenizer.encode("he") + [tokenizer.eos_id]
        decoded = tokenizer.decode(ids_with_special)

        assert decoded == "he"


class TestTextDataset:
    """Tests for TextDataset class."""

    def test_init(self):
        """Test TextDataset initialization."""
        tokenizer = SimpleTokenizer()
        text = "hello world"
        tokenizer.build_vocab([text])

        dataset = TextDataset(text, tokenizer, seq_len=5)

        assert dataset.seq_len == 5
        assert len(dataset.tokens) == len(text)

    def test_len(self):
        """Test dataset length calculation."""
        tokenizer = SimpleTokenizer()
        text = "hello world"
        tokenizer.build_vocab([text])
        seq_len = 5

        dataset = TextDataset(text, tokenizer, seq_len=seq_len)

        expected_len = len(text) - seq_len
        assert len(dataset) == expected_len

    def test_getitem(self):
        """Test getting a single item from dataset."""
        tokenizer = SimpleTokenizer()
        text = "hello world"
        tokenizer.build_vocab([text])
        seq_len = 5

        dataset = TextDataset(text, tokenizer, seq_len=seq_len)
        input_ids, target_ids = dataset[0]

        assert input_ids.shape == (seq_len,)
        assert target_ids.shape == (seq_len,)
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long

    def test_input_target_shift(self):
        """Test that target is shifted by 1 from input."""
        tokenizer = SimpleTokenizer()
        text = "abcdefghij"
        tokenizer.build_vocab([text])
        seq_len = 5

        dataset = TextDataset(text, tokenizer, seq_len=seq_len)
        input_ids, target_ids = dataset[0]

        assert torch.equal(input_ids[1:], target_ids[:-1])

    def test_empty_dataset(self):
        """Test dataset with text shorter than seq_len."""
        tokenizer = SimpleTokenizer()
        text = "hi"
        tokenizer.build_vocab([text])
        seq_len = 10

        dataset = TextDataset(text, tokenizer, seq_len=seq_len)

        assert len(dataset) == 0


class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_train_epoch_returns_loss(self):
        """Test that train_epoch returns a loss value."""
        tokenizer = SimpleTokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokenizer.build_vocab([text])

        dataset = TextDataset(text, tokenizer, seq_len=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')

        loss = train_epoch(model, dataloader, optimizer, device, epoch=0)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_decreases_loss(self):
        """Test that training decreases loss over multiple epochs."""
        tokenizer = SimpleTokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 20
        tokenizer.build_vocab([text])

        dataset = TextDataset(text, tokenizer, seq_len=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device('cpu')

        loss1 = train_epoch(model, dataloader, optimizer, device, epoch=0)
        loss2 = train_epoch(model, dataloader, optimizer, device, epoch=1)
        loss3 = train_epoch(model, dataloader, optimizer, device, epoch=2)

        assert loss3 < loss1


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_returns_loss(self):
        """Test that evaluate returns a loss value."""
        tokenizer = SimpleTokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokenizer.build_vocab([text])

        dataset = TextDataset(text, tokenizer, seq_len=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
        )

        device = torch.device('cpu')

        loss = evaluate(model, dataloader, device)

        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate_no_gradients(self):
        """Test that evaluate does not compute gradients."""
        tokenizer = SimpleTokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokenizer.build_vocab([text])

        dataset = TextDataset(text, tokenizer, seq_len=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
        )

        device = torch.device('cpu')

        for param in model.parameters():
            param.grad = None

        evaluate(model, dataloader, device)

        for param in model.parameters():
            assert param.grad is None

    def test_evaluate_deterministic(self):
        """Test that evaluate produces same result on same data."""
        tokenizer = SimpleTokenizer()
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokenizer.build_vocab([text])

        dataset = TextDataset(text, tokenizer, seq_len=16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
        )

        device = torch.device('cpu')

        loss1 = evaluate(model, dataloader, device)
        loss2 = evaluate(model, dataloader, device)

        assert loss1 == loss2


class TestIntegration:
    """Integration tests for training pipeline."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        tokenizer = SimpleTokenizer()
        text = "Hello world. This is a test. Machine learning is great. " * 20
        tokenizer.build_vocab([text])

        train_text = text[:int(len(text) * 0.8)]
        val_text = text[int(len(text) * 0.8):]

        train_dataset = TextDataset(train_text, tokenizer, seq_len=16)
        val_dataset = TextDataset(val_text, tokenizer, seq_len=16)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=32,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device('cpu')

        initial_val_loss = evaluate(model, val_loader, device)

        for epoch in range(3):
            train_epoch(model, train_loader, optimizer, device, epoch)

        final_val_loss = evaluate(model, val_loader, device)

        assert final_val_loss < initial_val_loss


class TestWithTestData:
    """Tests using test data files."""

    def test_load_sample_text(self):
        """Test loading sample text from test data."""
        test_data_path = os.path.join(
            os.path.dirname(__file__), "data", "sample_text.txt"
        )

        if os.path.exists(test_data_path):
            with open(test_data_path, "r") as f:
                text = f.read()

            tokenizer = SimpleTokenizer()
            tokenizer.build_vocab([text])

            assert tokenizer.vocab_size > 0
            assert len(tokenizer.encode(text)) == len(text)

    def test_load_test_config(self):
        """Test loading test configuration."""
        config_path = os.path.join(
            os.path.dirname(__file__), "data", "test_config.json"
        )

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

            assert "model_config" in config
            assert "training_config" in config
            assert "test_prompts" in config

            model_config = config["model_config"]
            assert "d_model" in model_config
            assert "n_heads" in model_config
            assert "n_layers" in model_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
