# TinyLLM: Educational Transformer Language Model

**日本語** | [English](#english-version)

---

## 日本語版

### 概要

TinyLLMは、Transformerアーキテクチャを使用した言語モデルの学習用実装です。パフォーマンスよりも**理解しやすさ**を重視して設計されています。

### 🎯 このプロジェクトの目的

1. **Transformerの内部構造を理解する**
2. **言語モデルの学習方法を学ぶ**
3. **テキスト生成（推論）の仕組みを理解する**

---

### 📚 Transformerとは？

Transformerは、2017年に「Attention Is All You Need」論文で提案されたニューラルネットワークアーキテクチャです。GPT、BERT、ChatGPTなど、現代の大規模言語モデル（LLM）の基盤となっています。

#### Transformerの主要コンポーネント

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Model                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  入力トークン: [私, は, 猫, が, 好き]                          │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Token Embedding（トークン埋め込み）               │   │
│  │    各トークンを高次元ベクトルに変換                     │   │
│  │    例: "猫" → [0.1, -0.3, 0.5, 0.2, ...]           │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. Positional Encoding（位置エンコーディング）        │   │
│  │    単語の位置情報を埋め込みに追加                      │   │
│  │    "私は猫" ≠ "猫は私" を区別するために必要           │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. Transformer Block ×N                             │   │
│  │    ┌───────────────────────────────────────────┐   │   │
│  │    │ Self-Attention（自己注意機構）             │   │   │
│  │    │ 各単語が他の全単語との関係を学習            │   │   │
│  │    └───────────────────────────────────────────┘   │   │
│  │              ↓                                      │   │
│  │    ┌───────────────────────────────────────────┐   │   │
│  │    │ Feed Forward Network（全結合層）          │   │   │
│  │    │ 非線形変換を適用                           │   │   │
│  │    └───────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. Output Layer（出力層）                           │   │
│  │    次のトークンの確率分布を出力                       │   │
│  │    P(です)=0.3, P(だ)=0.2, P(を)=0.15, ...        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 🔍 Self-Attention（自己注意機構）の解説

Self-Attentionは、Transformerの心臓部です。各トークンが他の全トークンとの関係性を計算します。

#### Query, Key, Value（Q, K, V）

```
入力: "猫がマットの上で寝ている"

各トークンから3つのベクトルを生成:
- Query (Q): 「何を探しているか」
- Key (K): 「何を持っているか」
- Value (V): 「実際の情報」

注意スコアの計算:
Score(猫, マット) = Q(猫) · K(マット) / √d_k

「猫」は「マット」「寝ている」に強く注意を向ける
→ 文脈を理解するために重要な単語を識別
```

#### 計算式

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

1. Q × K^T: 各トークン間の類似度を計算
2. / √d_k: スケーリング（勾配の安定化）
3. softmax: 確率分布に変換
4. × V: 重み付き和を計算
```

#### 因果マスキング（Causal Masking）

言語モデルでは、未来のトークンを見ることは「カンニング」になります。

```
入力: [私, は, 猫, が, 好き]

注意マスク:
        私  は  猫  が  好き
私    [ 1   0   0   0   0  ]  ← 私は自分自身のみ見れる
は    [ 1   1   0   0   0  ]  ← は は私と自分を見れる
猫    [ 1   1   1   0   0  ]  ← 猫は私、は、自分を見れる
が    [ 1   1   1   1   0  ]
好き  [ 1   1   1   1   1  ]  ← 好きは全部見れる
```

---

### 📖 学習プロセス

#### 学習の目標：次のトークン予測

```
入力:    [私] [は] [猫] [が]
ターゲット: [は] [猫] [が] [好き]

モデルは各位置で次の単語を予測することを学習します
```

#### 学習ループ

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. フォワードパス：予測を計算
        predictions = model(input_tokens)

        # 2. 損失計算：予測とターゲットを比較
        loss = cross_entropy(predictions, target_tokens)

        # 3. バックワードパス：勾配を計算
        loss.backward()

        # 4. パラメータ更新：重みを調整
        optimizer.step()
```

#### 損失関数：Cross-Entropy Loss

```
予測確率: P(猫)=0.7, P(犬)=0.2, P(鳥)=0.1
正解: 猫

Loss = -log(P(猫)) = -log(0.7) ≈ 0.36

正解の確率が高いほど、損失は小さくなる
```

---

### 🚀 推論（テキスト生成）

#### 自己回帰生成

```
ステップ1: 入力 = [私]
         → 予測: P(は)=0.4, P(が)=0.3, ...
         → サンプリング: [は]

ステップ2: 入力 = [私, は]
         → 予測: P(猫)=0.3, P(学生)=0.2, ...
         → サンプリング: [猫]

ステップ3: 入力 = [私, は, 猫]
         → 予測: P(が)=0.4, P(を)=0.3, ...
         → サンプリング: [が]

... 繰り返し ...

最終出力: [私, は, 猫, が, 好き, です]
```

#### サンプリング手法

| 手法 | 説明 | 用途 |
|------|------|------|
| Greedy | 最も確率の高いトークンを選択 | 決定論的な出力 |
| Temperature | 確率分布を調整 | 創造性の制御 |
| Top-K | 上位K個から選択 | 多様性の制限 |
| Top-P (Nucleus) | 累積確率P以内から選択 | 動的な候補数 |

---

### 🛠️ 使い方

#### インストール

```bash
# 必要なライブラリ
pip install torch
```

#### 学習

```bash
python train.py
```

#### 推論

```bash
# 単発生成
python inference.py --model tiny_llm_model.pt --prompt "Hello"

# 対話モード
python inference.py --model tiny_llm_model.pt --interactive

# サンプリング手法のデモ
python inference.py --model tiny_llm_model.pt --demo
```

#### デモ実行

```bash
# モデルのデモ
python tiny_transformer.py
```

---

### 📁 ファイル構成

```
TinyLLM/
├── tiny_transformer.py  # Transformerモデルの実装
├── train.py             # 学習スクリプト
├── inference.py         # 推論スクリプト
└── README.md            # このファイル
```

---

### 🔧 モデル設定

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `vocab_size` | 語彙サイズ | - |
| `d_model` | 埋め込み次元 | 256 |
| `n_heads` | 注意ヘッド数 | 4 |
| `n_layers` | Transformer層数 | 4 |
| `d_ff` | FFN次元 | 1024 |
| `max_seq_len` | 最大系列長 | 512 |
| `dropout` | ドロップアウト率 | 0.1 |

---

### 📚 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原論文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 視覚的解説
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

<a name="english-version"></a>

## English Version

### Overview

TinyLLM is an educational implementation of a language model using the Transformer architecture. It is designed for **clarity and understanding** rather than performance.

### 🎯 Purpose of This Project

1. **Understand the internal structure of Transformers**
2. **Learn how language models are trained**
3. **Understand how text generation (inference) works**

---

### 📚 What is a Transformer?

The Transformer is a neural network architecture proposed in the 2017 paper "Attention Is All You Need". It forms the foundation of modern large language models (LLMs) like GPT, BERT, and ChatGPT.

#### Main Components of a Transformer

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Model                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input tokens: [The, cat, sat, on, the, mat]                │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Token Embedding                                   │   │
│  │    Convert each token to a high-dimensional vector   │   │
│  │    e.g., "cat" → [0.1, -0.3, 0.5, 0.2, ...]        │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. Positional Encoding                               │   │
│  │    Add position information to embeddings            │   │
│  │    Needed to distinguish "cat sat" from "sat cat"   │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. Transformer Block ×N                             │   │
│  │    ┌───────────────────────────────────────────┐   │   │
│  │    │ Self-Attention                             │   │   │
│  │    │ Each word learns relationships with all    │   │   │
│  │    │ other words                                │   │   │
│  │    └───────────────────────────────────────────┘   │   │
│  │              ↓                                      │   │
│  │    ┌───────────────────────────────────────────┐   │   │
│  │    │ Feed Forward Network                       │   │   │
│  │    │ Apply non-linear transformations           │   │   │
│  │    └───────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. Output Layer                                     │   │
│  │    Output probability distribution for next token   │   │
│  │    P(is)=0.3, P(was)=0.2, P(the)=0.15, ...        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 🔍 Self-Attention Explained

Self-Attention is the heart of the Transformer. Each token computes its relationship with every other token.

#### Query, Key, Value (Q, K, V)

```
Input: "The cat sat on the mat"

Three vectors are generated from each token:
- Query (Q): "What am I looking for?"
- Key (K): "What do I have?"
- Value (V): "The actual information"

Attention score calculation:
Score(cat, mat) = Q(cat) · K(mat) / √d_k

"cat" strongly attends to "mat" and "sat"
→ Identifies important words for understanding context
```

#### Formula

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

1. Q × K^T: Calculate similarity between each token pair
2. / √d_k: Scaling (stabilizes gradients)
3. softmax: Convert to probability distribution
4. × V: Compute weighted sum
```

#### Causal Masking

In language models, looking at future tokens would be "cheating".

```
Input: [The, cat, sat, on, the]

Attention mask:
         The  cat  sat  on  the
The    [  1    0    0   0    0 ]  ← "The" can only see itself
cat    [  1    1    0   0    0 ]  ← "cat" can see "The" and itself
sat    [  1    1    1   0    0 ]  ← "sat" can see "The", "cat", itself
on     [  1    1    1   1    0 ]
the    [  1    1    1   1    1 ]  ← "the" can see everything before it
```

---

### 📖 Training Process

#### Training Objective: Next Token Prediction

```
Input:    [The] [cat] [sat] [on]
Target:   [cat] [sat] [on] [the]

The model learns to predict the next word at each position
```

#### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass: compute predictions
        predictions = model(input_tokens)

        # 2. Loss calculation: compare predictions with targets
        loss = cross_entropy(predictions, target_tokens)

        # 3. Backward pass: compute gradients
        loss.backward()

        # 4. Parameter update: adjust weights
        optimizer.step()
```

#### Loss Function: Cross-Entropy Loss

```
Predicted probabilities: P(cat)=0.7, P(dog)=0.2, P(bird)=0.1
Correct answer: cat

Loss = -log(P(cat)) = -log(0.7) ≈ 0.36

The higher the probability of the correct answer, the lower the loss
```

---

### 🚀 Inference (Text Generation)

#### Autoregressive Generation

```
Step 1: Input = [The]
        → Predict: P(cat)=0.4, P(dog)=0.3, ...
        → Sample: [cat]

Step 2: Input = [The, cat]
        → Predict: P(sat)=0.3, P(ran)=0.2, ...
        → Sample: [sat]

Step 3: Input = [The, cat, sat]
        → Predict: P(on)=0.4, P(down)=0.3, ...
        → Sample: [on]

... repeat ...

Final output: [The, cat, sat, on, the, mat]
```

#### Sampling Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Greedy | Select highest probability token | Deterministic output |
| Temperature | Adjust probability distribution | Control creativity |
| Top-K | Select from top K tokens | Limit diversity |
| Top-P (Nucleus) | Select from tokens within cumulative probability P | Dynamic candidate count |

---

### 🛠️ Usage

#### Installation

```bash
# Required libraries
pip install torch
```

#### Training

```bash
python train.py
```

#### Inference

```bash
# Single generation
python inference.py --model tiny_llm_model.pt --prompt "Hello"

# Interactive mode
python inference.py --model tiny_llm_model.pt --interactive

# Sampling methods demo
python inference.py --model tiny_llm_model.pt --demo
```

#### Demo Execution

```bash
# Model demo
python tiny_transformer.py
```

---

### 📁 File Structure

```
TinyLLM/
├── tiny_transformer.py  # Transformer model implementation
├── train.py             # Training script
├── inference.py         # Inference script
└── README.md            # This file
```

---

### 🔧 Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | - |
| `d_model` | Embedding dimension | 256 |
| `n_heads` | Number of attention heads | 4 |
| `n_layers` | Number of Transformer layers | 4 |
| `d_ff` | Feed-forward dimension | 1024 |
| `max_seq_len` | Maximum sequence length | 512 |
| `dropout` | Dropout rate | 0.1 |

---

### 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## License

MIT License
