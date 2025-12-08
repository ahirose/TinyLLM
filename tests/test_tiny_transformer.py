"""
Tests for TinyLLM Transformer components.
TinyLLM Transformerコンポーネントのテスト。
"""

import pytest
import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_transformer import (
    TokenEmbedding,
    PositionalEncoding,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerBlock,
    TinyLLM,
    count_parameters,
)


class TestTokenEmbedding:
    """Tests for TokenEmbedding class."""

    def test_init(self):
        """Test TokenEmbedding initialization."""
        vocab_size = 100
        d_model = 64
        embedding = TokenEmbedding(vocab_size, d_model)

        assert embedding.d_model == d_model
        assert embedding.embedding.num_embeddings == vocab_size
        assert embedding.embedding.embedding_dim == d_model

    def test_forward_shape(self):
        """Test TokenEmbedding forward pass output shape."""
        vocab_size = 100
        d_model = 64
        batch_size = 2
        seq_len = 10

        embedding = TokenEmbedding(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_scaling(self):
        """Test that embeddings are scaled by sqrt(d_model)."""
        vocab_size = 100
        d_model = 64

        embedding = TokenEmbedding(vocab_size, d_model)
        x = torch.tensor([[0]])

        raw_embedding = embedding.embedding(x)
        scaled_embedding = embedding(x)

        expected = raw_embedding * math.sqrt(d_model)
        assert torch.allclose(scaled_embedding, expected)


class TestPositionalEncoding:
    """Tests for PositionalEncoding class."""

    def test_init(self):
        """Test PositionalEncoding initialization."""
        d_model = 64
        max_seq_len = 128
        dropout = 0.1

        pe = PositionalEncoding(d_model, max_seq_len, dropout)

        assert pe.pe.shape == (1, max_seq_len, d_model)
        assert pe.dropout.p == dropout

    def test_forward_shape(self):
        """Test PositionalEncoding forward pass output shape."""
        d_model = 64
        max_seq_len = 128
        batch_size = 2
        seq_len = 10

        pe = PositionalEncoding(d_model, max_seq_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_values(self):
        """Test that positional encoding values follow sin/cos pattern."""
        d_model = 64
        max_seq_len = 128

        pe = PositionalEncoding(d_model, max_seq_len, dropout=0.0)

        assert pe.pe[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert pe.pe[0, 0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_different_positions_have_different_encodings(self):
        """Test that different positions have different encodings."""
        d_model = 64
        max_seq_len = 128

        pe = PositionalEncoding(d_model, max_seq_len, dropout=0.0)

        assert not torch.allclose(pe.pe[0, 0], pe.pe[0, 1])
        assert not torch.allclose(pe.pe[0, 0], pe.pe[0, 10])


class TestMultiHeadSelfAttention:
    """Tests for MultiHeadSelfAttention class."""

    def test_init(self):
        """Test MultiHeadSelfAttention initialization."""
        d_model = 64
        n_heads = 4
        dropout = 0.1

        attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        assert attention.d_model == d_model
        assert attention.n_heads == n_heads
        assert attention.d_k == d_model // n_heads

    def test_init_invalid_heads(self):
        """Test that initialization fails when d_model is not divisible by n_heads."""
        d_model = 64
        n_heads = 5

        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(d_model, n_heads)

    def test_forward_shape(self):
        """Test MultiHeadSelfAttention forward pass output shape."""
        d_model = 64
        n_heads = 4
        batch_size = 2
        seq_len = 10

        attention = MultiHeadSelfAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_with_mask(self):
        """Test MultiHeadSelfAttention forward pass with mask."""
        d_model = 64
        n_heads = 4
        batch_size = 2
        seq_len = 10

        attention = MultiHeadSelfAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = attention(x, mask)

        assert output.shape == (batch_size, seq_len, d_model)


class TestFeedForward:
    """Tests for FeedForward class."""

    def test_init(self):
        """Test FeedForward initialization."""
        d_model = 64
        d_ff = 256
        dropout = 0.1

        ff = FeedForward(d_model, d_ff, dropout)

        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model

    def test_forward_shape(self):
        """Test FeedForward forward pass output shape."""
        d_model = 64
        d_ff = 256
        batch_size = 2
        seq_len = 10

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerBlock:
    """Tests for TransformerBlock class."""

    def test_init(self):
        """Test TransformerBlock initialization."""
        d_model = 64
        n_heads = 4
        d_ff = 256
        dropout = 0.1

        block = TransformerBlock(d_model, n_heads, d_ff, dropout)

        assert isinstance(block.attention, MultiHeadSelfAttention)
        assert isinstance(block.feed_forward, FeedForward)
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)

    def test_forward_shape(self):
        """Test TransformerBlock forward pass output shape."""
        d_model = 64
        n_heads = 4
        d_ff = 256
        batch_size = 2
        seq_len = 10

        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_with_mask(self):
        """Test TransformerBlock forward pass with mask."""
        d_model = 64
        n_heads = 4
        d_ff = 256
        batch_size = 2
        seq_len = 10

        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = block(x, mask)

        assert output.shape == (batch_size, seq_len, d_model)


class TestTinyLLM:
    """Tests for TinyLLM class."""

    def test_init(self):
        """Test TinyLLM initialization."""
        vocab_size = 100
        d_model = 64
        n_heads = 4
        n_layers = 2
        d_ff = 256
        max_seq_len = 128

        model = TinyLLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        )

        assert model.vocab_size == vocab_size
        assert model.d_model == d_model
        assert model.max_seq_len == max_seq_len
        assert len(model.layers) == n_layers

    def test_forward_shape(self):
        """Test TinyLLM forward pass output shape."""
        vocab_size = 100
        d_model = 64
        n_heads = 4
        n_layers = 2
        batch_size = 2
        seq_len = 10

        model = TinyLLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = model(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        vocab_size = 100
        model = TinyLLM(vocab_size=vocab_size)

        seq_len = 4
        mask = model.create_causal_mask(seq_len, torch.device('cpu'))

        expected = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ], dtype=torch.float).unsqueeze(0).unsqueeze(0)

        assert torch.allclose(mask, expected)

    def test_generate(self):
        """Test text generation."""
        vocab_size = 100
        d_model = 64
        n_heads = 4
        n_layers = 2

        model = TinyLLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        prompt = torch.randint(0, vocab_size, (1, 5))
        max_new_tokens = 10

        generated = model.generate(prompt, max_new_tokens=max_new_tokens)

        assert generated.shape == (1, 5 + max_new_tokens)

    def test_generate_with_temperature(self):
        """Test text generation with different temperatures."""
        vocab_size = 100
        model = TinyLLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        prompt = torch.randint(0, vocab_size, (1, 5))

        generated_low_temp = model.generate(prompt, max_new_tokens=5, temperature=0.1)
        generated_high_temp = model.generate(prompt, max_new_tokens=5, temperature=2.0)

        assert generated_low_temp.shape == (1, 10)
        assert generated_high_temp.shape == (1, 10)

    def test_generate_with_top_k(self):
        """Test text generation with top-k sampling."""
        vocab_size = 100
        model = TinyLLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        prompt = torch.randint(0, vocab_size, (1, 5))

        generated = model.generate(prompt, max_new_tokens=5, top_k=10)

        assert generated.shape == (1, 10)

    def test_generate_with_top_p(self):
        """Test text generation with top-p (nucleus) sampling."""
        vocab_size = 100
        model = TinyLLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        prompt = torch.randint(0, vocab_size, (1, 5))

        generated = model.generate(prompt, max_new_tokens=5, top_p=0.9)

        assert generated.shape == (1, 10)


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters(self):
        """Test parameter counting."""
        vocab_size = 100
        d_model = 64
        n_heads = 4
        n_layers = 2

        model = TinyLLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        param_count = count_parameters(model)

        expected_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == expected_count
        assert param_count > 0


class TestModelGradients:
    """Tests for gradient flow through the model."""

    def test_gradients_flow(self):
        """Test that gradients flow through all layers."""
        vocab_size = 100
        model = TinyLLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        x = torch.randint(0, vocab_size, (2, 10))
        output = model(x)

        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestModelDeterminism:
    """Tests for model determinism."""

    def test_eval_mode_determinism(self):
        """Test that model produces same output in eval mode."""
        vocab_size = 100
        model = TinyLLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)
        model.eval()

        x = torch.randint(0, vocab_size, (2, 10))

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
