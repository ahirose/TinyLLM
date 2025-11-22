"""
TinyLLM: A Simple Transformer-based Language Model for Educational Purposes
TinyLLM: 教育目的のためのシンプルなTransformerベース言語モデル

This implementation focuses on clarity and understanding rather than performance.
この実装は、パフォーマンスよりも明確さと理解しやすさを重視しています。

Architecture Overview / アーキテクチャ概要:
┌─────────────────────────────────────────────────────────────────┐
│                        TinyLLM Model                            │
├─────────────────────────────────────────────────────────────────┤
│  Input: Token IDs (e.g., [1, 42, 17, 8])                       │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Token Embedding: 単語をベクトルに変換                    │   │
│  │    Convert tokens to dense vectors                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. Positional Encoding: 位置情報を追加                     │   │
│  │    Add position information to embeddings                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. Transformer Blocks (×N layers):                       │   │
│  │    ┌───────────────────────────────────────────────┐     │   │
│  │    │ 3a. Multi-Head Self-Attention                 │     │   │
│  │    │     自己注意機構：各単語が他の単語を参照           │     │   │
│  │    └───────────────────────────────────────────────┘     │   │
│  │                        ↓                                 │   │
│  │    ┌───────────────────────────────────────────────┐     │   │
│  │    │ 3b. Feed Forward Network                      │     │   │
│  │    │     全結合層：非線形変換を適用                     │     │   │
│  │    └───────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. Output Layer: 次の単語の確率分布を出力                  │   │
│  │    Project to vocabulary size for next token prediction  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  Output: Logits for each vocabulary token                      │
└─────────────────────────────────────────────────────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer / トークン埋め込み層

    Purpose / 目的:
    - Converts discrete token IDs into continuous vector representations
    - 離散的なトークンIDを連続的なベクトル表現に変換する

    Example / 例:
    - Token ID 42 ("hello") → [0.1, -0.3, 0.5, ...] (d_model dimensional vector)
    - トークンID 42 ("hello") → [0.1, -0.3, 0.5, ...] (d_model次元のベクトル)
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary / 語彙のサイズ
            d_model: Dimension of embeddings / 埋め込みの次元数
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape (batch_size, seq_len)
               トークンID、形状 (バッチサイズ, 系列長)

        Returns:
            Embeddings, shape (batch_size, seq_len, d_model)
            埋め込み、形状 (バッチサイズ, 系列長, d_model)
        """
        # Scale embeddings by sqrt(d_model) for stable training
        # 学習の安定化のため、sqrt(d_model)でスケーリング
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding / 位置エンコーディング

    Purpose / 目的:
    - Adds position information to embeddings
    - 埋め込みに位置情報を追加する

    Why needed / なぜ必要か:
    - Self-attention treats all positions equally
    - 自己注意機構は全ての位置を平等に扱う
    - Without positional encoding, "I love you" = "you love I"
    - 位置エンコーディングがないと、「私は君を愛する」=「君は私を愛する」となってしまう

    Method / 手法:
    - Uses sine and cosine functions of different frequencies
    - 異なる周波数のsin関数とcos関数を使用
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of embeddings / 埋め込みの次元数
            max_seq_len: Maximum sequence length / 最大系列長
            dropout: Dropout rate / ドロップアウト率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        # 位置エンコーディング行列を作成
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the formula
        # 式の分母を計算
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        # 偶数インデックスにsin、奇数インデックスにcosを適用
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        # バッチ次元を追加し、バッファとして登録（パラメータではない）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings, shape (batch_size, seq_len, d_model)
               入力埋め込み、形状 (バッチサイズ, 系列長, d_model)

        Returns:
            Embeddings with positional information added
            位置情報が追加された埋め込み
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention / マルチヘッド自己注意機構

    Purpose / 目的:
    - Allows each token to "look at" other tokens in the sequence
    - 各トークンがシーケンス内の他のトークンを「見る」ことを可能にする

    How it works / 仕組み:
    1. Create Query (Q), Key (K), Value (V) from input
       入力からQuery (Q), Key (K), Value (V)を作成
    2. Calculate attention scores: how much each token should attend to others
       注意スコアを計算：各トークンが他のトークンにどれだけ注目すべきか
    3. Weighted sum of Values based on attention scores
       注意スコアに基づいてValueの加重和を計算

    Multi-head / マルチヘッド:
    - Split attention into multiple "heads" that learn different patterns
    - 注意機構を複数の「ヘッド」に分割し、異なるパターンを学習

    Causal Masking / 因果マスキング:
    - For language models, tokens can only attend to previous tokens
    - 言語モデルでは、トークンは前のトークンにのみ注目できる
    - Prevents "cheating" by looking at future tokens
    - 未来のトークンを見ることによる「カンニング」を防ぐ

    Attention Formula / 注意機構の式:
    Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension / モデルの次元数
            n_heads: Number of attention heads / 注意ヘッドの数
            dropout: Dropout rate / ドロップアウト率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head / 各ヘッドの次元

        # Linear projections for Q, K, V
        # Q, K, Vのための線形変換
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        # 出力変換
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
               入力テンソル、形状 (バッチサイズ, 系列長, d_model)
            mask: Optional attention mask / オプションの注意マスク

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
            出力テンソル、形状 (バッチサイズ, 系列長, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections to get Q, K, V
        # ステップ1：線形変換でQ, K, Vを取得
        q = self.w_q(x)  # (batch_size, seq_len, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # Step 2: Reshape for multi-head attention
        # ステップ2：マルチヘッド注意のために形状を変更
        # (batch_size, seq_len, d_model) → (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Calculate attention scores
        # ステップ3：注意スコアを計算
        # scores = Q × K^T / √d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch_size, n_heads, seq_len, seq_len)

        # Step 4: Apply causal mask (for language modeling)
        # ステップ4：因果マスクを適用（言語モデリング用）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 5: Apply softmax to get attention weights
        # ステップ5：ソフトマックスで注意重みを取得
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 6: Weighted sum of values
        # ステップ6：値の加重和
        # output = attention_weights × V
        output = torch.matmul(attention_weights, v)
        # output shape: (batch_size, n_heads, seq_len, d_k)

        # Step 7: Concatenate heads and project
        # ステップ7：ヘッドを結合して投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        return output


class FeedForward(nn.Module):
    """
    Feed Forward Network / フィードフォワードネットワーク

    Purpose / 目的:
    - Applies non-linear transformation to each position independently
    - 各位置に独立して非線形変換を適用

    Structure / 構造:
    - Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)
    - 線形変換(d_model → d_ff) → ReLU → 線形変換(d_ff → d_model)

    Why needed / なぜ必要か:
    - Self-attention is mostly linear, FFN adds non-linearity
    - 自己注意機構はほぼ線形なので、FFNが非線形性を追加
    - Allows the model to learn complex patterns
    - モデルが複雑なパターンを学習することを可能にする
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension / モデルの次元数
            d_ff: Feed-forward dimension (usually 4 * d_model)
                  フィードフォワード次元（通常は4 * d_model）
            dropout: Dropout rate / ドロップアウト率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
               入力テンソル、形状 (バッチサイズ, 系列長, d_model)

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
            出力テンソル、形状 (バッチサイズ, 系列長, d_model)
        """
        # Expand → ReLU → Contract
        # 拡大 → ReLU → 縮小
        x = self.linear1(x)      # (batch_size, seq_len, d_ff)
        x = F.relu(x)            # Non-linear activation / 非線形活性化
        x = self.dropout(x)
        x = self.linear2(x)      # (batch_size, seq_len, d_model)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block / Transformerブロック

    Structure / 構造:
    ┌────────────────────────────────────────┐
    │ Input                                  │
    │   ↓                                    │
    │ LayerNorm → Self-Attention → + (residual)
    │   ↓                                    │
    │ LayerNorm → FeedForward → + (residual) │
    │   ↓                                    │
    │ Output                                 │
    └────────────────────────────────────────┘

    Key concepts / 重要な概念:

    1. Residual Connection / 残差接続:
       - output = input + sublayer(input)
       - Helps gradient flow in deep networks
       - 深いネットワークでの勾配の流れを助ける

    2. Layer Normalization / 層正規化:
       - Normalizes activations for stable training
       - 学習の安定化のために活性化を正規化
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension / モデルの次元数
            n_heads: Number of attention heads / 注意ヘッドの数
            d_ff: Feed-forward dimension / フィードフォワード次元
            dropout: Dropout rate / ドロップアウト率
        """
        super().__init__()

        # Sub-layers
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
               入力テンソル、形状 (バッチサイズ, 系列長, d_model)
            mask: Attention mask / 注意マスク

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
            出力テンソル、形状 (バッチサイズ, 系列長, d_model)
        """
        # Self-attention with residual connection
        # 残差接続付き自己注意
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        # 残差接続付きフィードフォワード
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class TinyLLM(nn.Module):
    """
    TinyLLM: A Simple Transformer Language Model
    TinyLLM: シンプルなTransformer言語モデル

    This is the main model class that combines all components.
    これは全てのコンポーネントを組み合わせたメインモデルクラスです。

    Training Objective / 学習目標:
    - Next token prediction: Given tokens [1, 2, 3], predict token [4]
    - 次のトークン予測：トークン [1, 2, 3] が与えられたら、トークン [4] を予測

    Loss Function / 損失関数:
    - Cross-entropy loss between predicted and actual next tokens
    - 予測された次のトークンと実際のトークンの交差エントロピー損失
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of vocabulary / 語彙のサイズ
            d_model: Model dimension (default: 256) / モデルの次元数
            n_heads: Number of attention heads (default: 4) / 注意ヘッドの数
            n_layers: Number of transformer layers (default: 4) / Transformer層の数
            d_ff: Feed-forward dimension (default: 1024) / フィードフォワード次元
            max_seq_len: Maximum sequence length (default: 512) / 最大系列長
            dropout: Dropout rate (default: 0.1) / ドロップアウト率
        """
        super().__init__()

        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding layers
        # 埋め込み層
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer blocks
        # Transformerブロック
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        # 最終層正規化
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (to vocabulary size)
        # 出力投影（語彙サイズへ）
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        # 重みの初期化
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        Xavier/Glorot初期化を使用して重みを初期化。

        Good initialization helps with:
        良い初期化は以下を助ける：
        - Faster convergence / より速い収束
        - Better final performance / より良い最終性能
        - Avoiding vanishing/exploding gradients / 勾配消失・爆発の回避
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal (look-ahead) mask for self-attention.
        自己注意のための因果（先読み）マスクを作成。

        Example for seq_len=4 / seq_len=4の例:
        [[1, 0, 0, 0],   Position 0 can only see position 0
         [1, 1, 0, 0],   Position 1 can see positions 0, 1
         [1, 1, 1, 0],   Position 2 can see positions 0, 1, 2
         [1, 1, 1, 1]]   Position 3 can see positions 0, 1, 2, 3

        This prevents the model from "cheating" by looking at future tokens.
        これにより、モデルが未来のトークンを見て「カンニング」することを防ぐ。
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        モデルのフォワードパス。

        Args:
            x: Input token IDs, shape (batch_size, seq_len)
               入力トークンID、形状 (バッチサイズ, 系列長)

        Returns:
            Logits for next token prediction, shape (batch_size, seq_len, vocab_size)
            次のトークン予測のためのロジット、形状 (バッチサイズ, 系列長, vocab_size)
        """
        seq_len = x.size(1)
        device = x.device

        # Step 1: Token embedding
        # ステップ1：トークン埋め込み
        x = self.token_embedding(x)

        # Step 2: Add positional encoding
        # ステップ2：位置エンコーディングを追加
        x = self.positional_encoding(x)

        # Step 3: Create causal mask
        # ステップ3：因果マスクを作成
        mask = self.create_causal_mask(seq_len, device)

        # Step 4: Pass through transformer blocks
        # ステップ4：Transformerブロックを通過
        for layer in self.layers:
            x = layer(x, mask)

        # Step 5: Final layer norm
        # ステップ5：最終層正規化
        x = self.final_norm(x)

        # Step 6: Project to vocabulary size
        # ステップ6：語彙サイズに投影
        logits = self.output_projection(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        自己回帰的にテキストを生成。

        Generation Process / 生成プロセス:
        1. Start with prompt tokens / プロンプトトークンから開始
        2. Predict next token / 次のトークンを予測
        3. Append predicted token / 予測したトークンを追加
        4. Repeat until max_new_tokens / max_new_tokensまで繰り返す

        Args:
            prompt_tokens: Starting tokens, shape (1, prompt_len)
                          開始トークン、形状 (1, プロンプト長)
            max_new_tokens: Maximum number of tokens to generate
                           生成する最大トークン数
            temperature: Sampling temperature (higher = more random)
                        サンプリング温度（高い = よりランダム）
            top_k: Keep only top k tokens for sampling
                  サンプリング時に上位kトークンのみを保持
            top_p: Keep tokens with cumulative probability < p (nucleus sampling)
                  累積確率がp未満のトークンを保持（核サンプリング）

        Returns:
            Generated tokens including prompt
            プロンプトを含む生成されたトークン
        """
        self.eval()
        generated = prompt_tokens

        for _ in range(max_new_tokens):
            # Truncate if exceeds max sequence length
            # 最大系列長を超えたら切り詰める
            if generated.size(1) > self.max_seq_len:
                context = generated[:, -self.max_seq_len:]
            else:
                context = generated

            # Get predictions
            # 予測を取得
            logits = self.forward(context)

            # Take logits for last position only
            # 最後の位置のロジットのみを取得
            logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            # 温度を適用
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            # top-kフィルタリングを適用
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            # top-p（核）フィルタリングを適用
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                # 閾値を超える累積確率のトークンを除去
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            # 分布からサンプリング
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            # 生成されたシーケンスに追加
            generated = torch.cat([generated, next_token], dim=1)

        return generated


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    モデルの学習可能なパラメータ数をカウント。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage / 使用例
if __name__ == "__main__":
    print("=" * 60)
    print("TinyLLM: Simple Transformer Language Model Demo")
    print("TinyLLM: シンプルなTransformer言語モデルのデモ")
    print("=" * 60)

    # Create model with small configuration
    # 小さな設定でモデルを作成
    model = TinyLLM(
        vocab_size=1000,    # Small vocabulary for demo / デモ用の小さな語彙
        d_model=128,        # Small embedding dimension / 小さな埋め込み次元
        n_heads=4,          # 4 attention heads / 4つの注意ヘッド
        n_layers=2,         # 2 transformer layers / 2つのTransformer層
        d_ff=512,           # Feed-forward dimension / フィードフォワード次元
        max_seq_len=128,    # Max sequence length / 最大系列長
        dropout=0.1,
    )

    print(f"\nModel Configuration / モデル設定:")
    print(f"  Vocabulary size / 語彙サイズ: {model.vocab_size}")
    print(f"  Model dimension / モデル次元: {model.d_model}")
    print(f"  Number of layers / 層の数: {len(model.layers)}")
    print(f"  Total parameters / 総パラメータ数: {count_parameters(model):,}")

    # Create dummy input
    # ダミー入力を作成
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"\nInput shape / 入力形状: {dummy_input.shape}")

    # Forward pass
    # フォワードパス
    output = model(dummy_input)
    print(f"Output shape / 出力形状: {output.shape}")

    # Generation demo
    # 生成デモ
    print("\n" + "=" * 60)
    print("Text Generation Demo / テキスト生成デモ")
    print("=" * 60)

    prompt = torch.randint(0, 1000, (1, 5))  # Random prompt
    print(f"Prompt tokens / プロンプトトークン: {prompt.tolist()}")

    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
    )
    print(f"Generated tokens / 生成されたトークン: {generated.tolist()}")

    print("\n" + "=" * 60)
    print("Demo complete! / デモ完了！")
    print("=" * 60)
