"""
Training Script for TinyLLM / TinyLLMの学習スクリプト

This script demonstrates how to train a Transformer language model.
このスクリプトはTransformer言語モデルの学習方法を示します。

Training Process Overview / 学習プロセスの概要:
┌─────────────────────────────────────────────────────────────────┐
│                    Training Loop / 学習ループ                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Prepare Data / データ準備                                    │
│     - Load text → Tokenize → Create batches                    │
│     - テキスト読込 → トークン化 → バッチ作成                       │
│                                                                 │
│  2. Forward Pass / フォワードパス                                 │
│     - Input tokens → Model → Predicted logits                  │
│     - 入力トークン → モデル → 予測ロジット                         │
│                                                                 │
│  3. Calculate Loss / 損失計算                                    │
│     - Compare predictions with actual next tokens              │
│     - 予測と実際の次のトークンを比較                               │
│     - Loss = CrossEntropy(predictions, targets)                │
│                                                                 │
│  4. Backward Pass / バックワードパス                              │
│     - Calculate gradients (∂Loss/∂weights)                     │
│     - 勾配を計算（∂Loss/∂weights）                               │
│                                                                 │
│  5. Update Weights / 重み更新                                    │
│     - weights = weights - learning_rate × gradients            │
│     - 重み = 重み - 学習率 × 勾配                                 │
│                                                                 │
│  6. Repeat / 繰り返し                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Key Concepts / 重要な概念:

1. Next Token Prediction / 次トークン予測:
   - Input:  "The cat sat on the"
   - Target: "cat sat on the mat"
   - Model learns to predict each next word
   - モデルは次の単語を予測することを学習

2. Batch Processing / バッチ処理:
   - Process multiple sequences at once for efficiency
   - 効率化のため複数のシーケンスを同時に処理

3. Gradient Descent / 勾配降下法:
   - Iteratively adjust weights to minimize loss
   - 損失を最小化するために重みを反復的に調整
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from typing import List, Optional
import os

from tiny_transformer import TinyLLM, count_parameters


class SimpleTokenizer:
    """
    A simple character-level tokenizer for educational purposes.
    教育目的のためのシンプルな文字レベルトークナイザー。

    In production LLMs, more sophisticated tokenizers like BPE are used.
    本番のLLMでは、BPEのようなより洗練されたトークナイザーが使用されます。

    Character-level tokenization is simpler to understand:
    文字レベルのトークン化はより理解しやすい：
    - "hello" → ['h', 'e', 'l', 'l', 'o'] → [104, 101, 108, 108, 111]
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

        # Special tokens / 特殊トークン
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Beginning of sequence / シーケンスの開始
        self.eos_token = "<EOS>"  # End of sequence / シーケンスの終了

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from texts.
        テキストから語彙を構築。

        Args:
            texts: List of training texts / 学習テキストのリスト
        """
        # Add special tokens first
        # 特殊トークンを最初に追加
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            self.char_to_id[token] = len(self.char_to_id)

        # Add all unique characters
        # 全てのユニークな文字を追加
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        for char in sorted(all_chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)

        # Create reverse mapping
        # 逆マッピングを作成
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

        print(f"Vocabulary built! Size: {self.vocab_size}")
        print(f"語彙を構築しました！サイズ: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        テキストをトークンIDに変換。

        Args:
            text: Input text / 入力テキスト

        Returns:
            List of token IDs / トークンIDのリスト
        """
        unk_id = self.char_to_id[self.unk_token]
        return [self.char_to_id.get(char, unk_id) for char in text]

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        トークンIDをテキストに戻す。

        Args:
            ids: List of token IDs / トークンIDのリスト

        Returns:
            Decoded text / デコードされたテキスト
        """
        chars = []
        for id in ids:
            if id in self.id_to_char:
                token = self.id_to_char[id]
                if token not in [self.pad_token, self.bos_token, self.eos_token]:
                    chars.append(token)
        return ''.join(chars)

    @property
    def pad_id(self):
        return self.char_to_id[self.pad_token]

    @property
    def bos_id(self):
        return self.char_to_id[self.bos_token]

    @property
    def eos_id(self):
        return self.char_to_id[self.eos_token]


class TextDataset(Dataset):
    """
    Dataset for language model training.
    言語モデル学習用のデータセット。

    Key concept: Creating input-target pairs for next token prediction
    重要な概念：次トークン予測のための入力-ターゲットペアの作成

    Example / 例:
    Text: "hello world"
    Tokens: [h, e, l, l, o, ' ', w, o, r, l, d]

    For seq_len=5:
    Input:  [h, e, l, l, o]      → Target: [e, l, l, o, ' ']
    Input:  [' ', w, o, r, l]    → Target: [w, o, r, l, d]

    The target is shifted by 1 position from input.
    ターゲットは入力から1位置シフトされます。
    """

    def __init__(self, text: str, tokenizer: SimpleTokenizer, seq_len: int):
        """
        Args:
            text: Training text / 学習テキスト
            tokenizer: Tokenizer to use / 使用するトークナイザー
            seq_len: Sequence length for each sample / 各サンプルの系列長
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Tokenize entire text
        # テキスト全体をトークン化
        self.tokens = tokenizer.encode(text)
        print(f"Total tokens: {len(self.tokens)}")
        print(f"総トークン数: {len(self.tokens)}")

    def __len__(self):
        # Number of sequences we can create
        # 作成できるシーケンスの数
        # Need seq_len + 1 tokens for each (input + 1 target token)
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        """
        Get a single training sample.
        単一の学習サンプルを取得。

        Returns:
            input_ids: Token IDs for input / 入力のトークンID
            target_ids: Token IDs for target (shifted by 1)
                       ターゲットのトークンID（1つシフト）
        """
        # Get seq_len + 1 tokens
        # seq_len + 1 トークンを取得
        chunk = self.tokens[idx:idx + self.seq_len + 1]

        # Input: first seq_len tokens
        # 入力：最初のseq_lenトークン
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)

        # Target: last seq_len tokens (shifted by 1)
        # ターゲット：最後のseq_lenトークン（1つシフト）
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.
    1エポック分モデルを学習。

    Args:
        model: The model to train / 学習するモデル
        dataloader: Training data loader / 学習データローダー
        optimizer: Optimizer / オプティマイザー
        device: Device to use / 使用するデバイス
        epoch: Current epoch number / 現在のエポック番号

    Returns:
        Average loss for the epoch / エポックの平均損失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        # Move data to device
        # データをデバイスに移動
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Step 1: Zero gradients
        # ステップ1：勾配をゼロに
        optimizer.zero_grad()

        # Step 2: Forward pass
        # ステップ2：フォワードパス
        logits = model(input_ids)
        # logits shape: (batch_size, seq_len, vocab_size)

        # Step 3: Calculate loss
        # ステップ3：損失を計算
        # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
        # 交差エントロピー用に形状変更: (batch_size * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        # Step 4: Backward pass
        # ステップ4：バックワードパス
        loss.backward()

        # Step 5: Clip gradients (prevents exploding gradients)
        # ステップ5：勾配クリッピング（勾配爆発を防ぐ）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 6: Update weights
        # ステップ6：重みを更新
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Print progress every 100 batches
        # 100バッチごとに進捗を表示
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the model on validation data.
    検証データでモデルを評価。

    Args:
        model: The model to evaluate / 評価するモデル
        dataloader: Validation data loader / 検証データローダー
        device: Device to use / 使用するデバイス

    Returns:
        Average loss / 平均損失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # No gradients needed for evaluation / 評価には勾配不要
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)

            loss = nn.functional.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def generate_sample(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.8,
) -> str:
    """
    Generate text from a prompt.
    プロンプトからテキストを生成。

    Args:
        model: Trained model / 学習済みモデル
        tokenizer: Tokenizer / トークナイザー
        prompt: Starting text / 開始テキスト
        max_new_tokens: Number of tokens to generate / 生成するトークン数
        device: Device to use / 使用するデバイス
        temperature: Sampling temperature / サンプリング温度

    Returns:
        Generated text / 生成されたテキスト
    """
    model.eval()

    # Encode prompt
    # プロンプトをエンコード
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Generate
    # 生成
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
    )

    # Decode
    # デコード
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


def train(
    training_text: str,
    model_config: dict = None,
    training_config: dict = None,
    save_path: str = "model.pt",
):
    """
    Main training function.
    メイン学習関数。

    Args:
        training_text: Text to train on / 学習するテキスト
        model_config: Model configuration / モデル設定
        training_config: Training configuration / 学習設定
        save_path: Path to save trained model / 学習済みモデルの保存パス
    """
    # Default configurations
    # デフォルト設定
    if model_config is None:
        model_config = {
            'd_model': 128,       # Embedding dimension / 埋め込み次元
            'n_heads': 4,         # Number of attention heads / 注意ヘッド数
            'n_layers': 4,        # Number of transformer layers / Transformer層数
            'd_ff': 512,          # Feed-forward dimension / フィードフォワード次元
            'max_seq_len': 256,   # Maximum sequence length / 最大系列長
            'dropout': 0.1,       # Dropout rate / ドロップアウト率
        }

    if training_config is None:
        training_config = {
            'batch_size': 32,
            'seq_len': 64,
            'epochs': 10,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
        }

    # Device selection
    # デバイス選択
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"使用デバイス: {device}")

    # Create tokenizer and build vocabulary
    # トークナイザーを作成し語彙を構築
    print("\n" + "=" * 60)
    print("Step 1: Building Vocabulary / ステップ1：語彙構築")
    print("=" * 60)
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([training_text])

    # Create dataset and dataloader
    # データセットとデータローダーを作成
    print("\n" + "=" * 60)
    print("Step 2: Creating Dataset / ステップ2：データセット作成")
    print("=" * 60)

    # Split into train/val (90/10)
    # 訓練/検証に分割 (90/10)
    split_idx = int(len(training_text) * 0.9)
    train_text = training_text[:split_idx]
    val_text = training_text[split_idx:]

    train_dataset = TextDataset(train_text, tokenizer, training_config['seq_len'])
    val_dataset = TextDataset(val_text, tokenizer, training_config['seq_len'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"訓練サンプル数: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"検証サンプル数: {len(val_dataset)}")

    # Create model
    # モデルを作成
    print("\n" + "=" * 60)
    print("Step 3: Creating Model / ステップ3：モデル作成")
    print("=" * 60)

    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        **model_config
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"モデルパラメータ数: {count_parameters(model):,}")

    # Create optimizer and scheduler
    # オプティマイザーとスケジューラーを作成
    print("\n" + "=" * 60)
    print("Step 4: Setting up Optimizer / ステップ4：オプティマイザー設定")
    print("=" * 60)

    optimizer = AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
    )

    # Learning rate scheduler (decreases LR over time)
    # 学習率スケジューラー（時間とともにLRを減少）
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config['epochs'],
    )

    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"学習率: {training_config['learning_rate']}")
    print(f"Weight decay: {training_config['weight_decay']}")
    print(f"重み減衰: {training_config['weight_decay']}")

    # Training loop
    # 学習ループ
    print("\n" + "=" * 60)
    print("Step 5: Training / ステップ5：学習")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(training_config['epochs']):
        start_time = time.time()

        print(f"\n--- Epoch {epoch + 1}/{training_config['epochs']} ---")
        print(f"--- エポック {epoch + 1}/{training_config['epochs']} ---")

        # Train
        # 学習
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        # 評価
        val_loss = evaluate(model, val_loader, device)

        # Update learning rate
        # 学習率を更新
        scheduler.step()

        elapsed = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  訓練損失: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  検証損失: {val_loss:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  所要時間: {elapsed:.1f}秒")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  学習率: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        # 最良モデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_char_to_id': tokenizer.char_to_id,
                'model_config': model_config,
            }, save_path)
            print(f"  Saved best model! / 最良モデルを保存しました！")

        # Generate sample text
        # サンプルテキストを生成
        if len(training_text) > 10:
            sample_prompt = training_text[:10]
            generated = generate_sample(
                model, tokenizer, sample_prompt, 50, device
            )
            print(f"  Sample generation / サンプル生成:")
            print(f"    Prompt: '{sample_prompt}'")
            print(f"    Output: '{generated[:100]}...'")

    print("\n" + "=" * 60)
    print("Training Complete! / 学習完了！")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"最良検証損失: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"モデル保存先: {save_path}")

    return model, tokenizer


# Example usage / 使用例
if __name__ == "__main__":
    print("=" * 60)
    print("TinyLLM Training Script")
    print("TinyLLM 学習スクリプト")
    print("=" * 60)

    # Sample training text (in practice, use a larger corpus)
    # サンプル学習テキスト（実際にはより大きなコーパスを使用）
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Neural networks are inspired by the human brain.
    Transformers have revolutionized natural language processing.
    Attention is all you need.
    Language models learn patterns from text data.
    Deep learning requires large amounts of data and compute.
    GPT models generate text one token at a time.
    The future of AI is exciting and full of possibilities.
    Programming is the art of solving problems with code.
    """ * 100  # Repeat to create more training data / 学習データを増やすために繰り返し

    # Train the model
    # モデルを学習
    model, tokenizer = train(
        training_text=sample_text,
        model_config={
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
            'max_seq_len': 128,
            'dropout': 0.1,
        },
        training_config={
            'batch_size': 16,
            'seq_len': 32,
            'epochs': 5,
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
        },
        save_path='tiny_llm_model.pt',
    )

    print("\nTraining complete!")
    print("学習が完了しました！")
