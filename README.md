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

### 📖 コードの読み方・お勧めの順番

Transformerと言語モデルを理解するために、以下の順番でコードを読むことをお勧めします。

#### ステップ1: `tiny_transformer.py` - モデルアーキテクチャ（最重要）

**読む順番:**

```
1. TinyLLM クラス (メインモデル)
   └── forward() メソッドで全体の流れを把握

2. TokenEmbedding クラス
   └── トークンIDを密なベクトルに変換する仕組み

3. PositionalEncoding クラス
   └── sin/cos関数による位置情報の埋め込み方法

4. MultiHeadSelfAttention クラス（★最重要）
   └── Q, K, V の計算
   └── Attention Score の計算
   └── Causal Mask の適用
   └── Multi-Head の結合

5. FeedForward クラス
   └── 2層の全結合ネットワーク
   └── GELU活性化関数

6. TransformerBlock クラス
   └── Attention → Add & Norm → FFN → Add & Norm の流れ
```

**ポイント:**
- 各クラスの `__init__` で構造を理解し、`forward` で処理の流れを追う
- コメントは日英両方あるので、理解しやすい方を読む
- `MultiHeadSelfAttention` が最も重要。時間をかけて理解する

#### ステップ2: `train.py` - 学習プロセス

**読む順番:**

```
1. SimpleTokenizer クラス
   └── 文字レベルトークナイザーの実装
   └── encode/decode の仕組み

2. TextDataset クラス
   └── 入力とターゲットのペア作成方法
   └── スライディングウィンドウ方式

3. train_epoch() 関数
   └── 1エポックの学習ループ
   └── forward → loss → backward → step の流れ

4. evaluate() 関数
   └── 検証データでの評価方法

5. train() 関数（メイン）
   └── 全体の学習フロー
   └── 学習率スケジューラー
   └── モデル保存のタイミング
```

**ポイント:**
- `train_epoch` の中のループが学習の本質
- 損失関数（CrossEntropyLoss）がどう使われるか確認
- 勾配計算とパラメータ更新の流れを追う

#### ステップ3: `inference.py` - テキスト生成

**読む順番:**

```
1. load_model() 関数
   └── 保存したモデルの読み込み方法

2. TinyLLM.generate() メソッド（tiny_transformer.py内）
   └── 自己回帰生成のループ
   └── Temperature による確率調整
   └── Top-K フィルタリング
   └── Top-P (Nucleus) サンプリング

3. generate_text() 関数
   └── 推論の全体フロー
   └── トークン化 → 生成 → デコード

4. demonstrate_sampling_methods() 関数
   └── 各サンプリング手法の比較
```

**ポイント:**
- `generate()` メソッドが自己回帰の核心
- サンプリング手法の違いを実際に試して理解する

#### 理解度チェックリスト

- [ ] Token Embedding が何をしているか説明できる
- [ ] Positional Encoding がなぜ必要か説明できる
- [ ] Q, K, V の役割を説明できる
- [ ] Attention Score の計算方法を説明できる
- [ ] Causal Mask がなぜ必要か説明できる
- [ ] Multi-Head Attention の利点を説明できる
- [ ] 学習時の損失関数の意味を説明できる
- [ ] 自己回帰生成の仕組みを説明できる
- [ ] Temperature の効果を説明できる

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

### 📖 How to Read the Code - Recommended Order

To understand Transformers and language models, we recommend reading the code in the following order.

#### Step 1: `tiny_transformer.py` - Model Architecture (Most Important)

**Reading order:**

```
1. TinyLLM class (main model)
   └── Understand the overall flow in forward() method

2. TokenEmbedding class
   └── How token IDs are converted to dense vectors

3. PositionalEncoding class
   └── Position information embedding using sin/cos functions

4. MultiHeadSelfAttention class (★ Most Important)
   └── Q, K, V computation
   └── Attention Score calculation
   └── Causal Mask application
   └── Multi-Head concatenation

5. FeedForward class
   └── 2-layer fully connected network
   └── GELU activation function

6. TransformerBlock class
   └── Flow: Attention → Add & Norm → FFN → Add & Norm
```

**Key points:**
- Understand the structure in each class's `__init__`, then follow the processing flow in `forward`
- Comments are in both Japanese and English - read whichever is easier for you
- `MultiHeadSelfAttention` is the most important. Take your time to understand it

#### Step 2: `train.py` - Training Process

**Reading order:**

```
1. SimpleTokenizer class
   └── Character-level tokenizer implementation
   └── encode/decode mechanism

2. TextDataset class
   └── How input-target pairs are created
   └── Sliding window approach

3. train_epoch() function
   └── Training loop for one epoch
   └── Flow: forward → loss → backward → step

4. evaluate() function
   └── How evaluation on validation data works

5. train() function (main)
   └── Overall training flow
   └── Learning rate scheduler
   └── When to save the model
```

**Key points:**
- The loop inside `train_epoch` is the essence of training
- Check how the loss function (CrossEntropyLoss) is used
- Follow the flow of gradient computation and parameter updates

#### Step 3: `inference.py` - Text Generation

**Reading order:**

```
1. load_model() function
   └── How to load a saved model

2. TinyLLM.generate() method (in tiny_transformer.py)
   └── Autoregressive generation loop
   └── Probability adjustment with Temperature
   └── Top-K filtering
   └── Top-P (Nucleus) sampling

3. generate_text() function
   └── Overall inference flow
   └── Tokenization → Generation → Decoding

4. demonstrate_sampling_methods() function
   └── Comparison of different sampling methods
```

**Key points:**
- The `generate()` method is the core of autoregressive generation
- Try different sampling methods to understand their differences

#### Understanding Checklist

- [ ] Can explain what Token Embedding does
- [ ] Can explain why Positional Encoding is necessary
- [ ] Can explain the roles of Q, K, V
- [ ] Can explain how Attention Score is calculated
- [ ] Can explain why Causal Mask is needed
- [ ] Can explain the benefits of Multi-Head Attention
- [ ] Can explain the meaning of the loss function during training
- [ ] Can explain how autoregressive generation works
- [ ] Can explain the effect of Temperature

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
