"""
Inference Script for TinyLLM / TinyLLMの推論スクリプト

This script demonstrates how to use a trained model for text generation.
このスクリプトは学習済みモデルを使用したテキスト生成方法を示します。

Inference Process Overview / 推論プロセスの概要:
┌─────────────────────────────────────────────────────────────────┐
│               Autoregressive Generation / 自己回帰生成            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Start with prompt tokens                               │
│  ステップ1：プロンプトトークンから開始                              │
│     [The] [cat] [sat]                                           │
│                                                                 │
│  Step 2: Forward pass → Get probabilities for next token        │
│  ステップ2：フォワードパス → 次のトークンの確率を取得               │
│     P(on)=0.3, P(down)=0.2, P(there)=0.15, ...                 │
│                                                                 │
│  Step 3: Sample next token based on probabilities               │
│  ステップ3：確率に基づいて次のトークンをサンプリング                 │
│     → Selected: [on]                                            │
│                                                                 │
│  Step 4: Append to sequence                                     │
│  ステップ4：シーケンスに追加                                       │
│     [The] [cat] [sat] [on]                                      │
│                                                                 │
│  Step 5: Repeat until done                                      │
│  ステップ5：完了まで繰り返し                                       │
│     [The] [cat] [sat] [on] [the] [mat]                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Sampling Methods / サンプリング手法:

1. Greedy (temperature=0):
   - Always select the most probable token
   - 常に最も確率の高いトークンを選択
   - Deterministic but may be repetitive
   - 決定論的だが繰り返しになりやすい

2. Temperature Sampling:
   - temperature < 1: More focused/deterministic
   - temperature < 1: より集中的/決定論的
   - temperature > 1: More random/creative
   - temperature > 1: よりランダム/創造的

3. Top-K Sampling:
   - Only consider top K most probable tokens
   - 上位K個の最も確率の高いトークンのみを考慮
   - Filters out unlikely tokens
   - 確率の低いトークンをフィルタリング

4. Top-P (Nucleus) Sampling:
   - Consider tokens until cumulative probability reaches P
   - 累積確率がPに達するまでのトークンを考慮
   - Dynamically adjusts number of candidates
   - 候補数を動的に調整
"""

import torch
import argparse
from typing import Optional
import os

from tiny_transformer import TinyLLM
from train import SimpleTokenizer


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple:
    """
    Load a trained model from disk.
    ディスクから学習済みモデルを読み込む。

    Args:
        model_path: Path to saved model / 保存されたモデルへのパス
        device: Device to load model on / モデルを読み込むデバイス

    Returns:
        Tuple of (model, tokenizer) / (モデル, トークナイザー)のタプル
    """
    print(f"Loading model from: {model_path}")
    print(f"モデルを読み込み中: {model_path}")

    # Load checkpoint
    # チェックポイントを読み込む
    checkpoint = torch.load(model_path, map_location=device)

    # Rebuild tokenizer
    # トークナイザーを再構築
    tokenizer = SimpleTokenizer()
    tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
    tokenizer.id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
    tokenizer.vocab_size = len(tokenizer.char_to_id)

    # Create and load model
    # モデルを作成して読み込む
    model_config = checkpoint['model_config']
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        **model_config
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"モデルの読み込みに成功しました！")
    print(f"  Vocabulary size / 語彙サイズ: {tokenizer.vocab_size}")
    print(f"  Model dimension / モデル次元: {model_config['d_model']}")
    print(f"  Number of layers / 層数: {model_config['n_layers']}")

    return model, tokenizer


@torch.no_grad()
def generate_text(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: torch.device = None,
    verbose: bool = True,
) -> str:
    """
    Generate text from a prompt.
    プロンプトからテキストを生成。

    This function demonstrates the inference process step by step.
    この関数は推論プロセスをステップごとに示します。

    Args:
        model: Trained model / 学習済みモデル
        tokenizer: Tokenizer / トークナイザー
        prompt: Starting text / 開始テキスト
        max_new_tokens: Maximum tokens to generate / 生成する最大トークン数
        temperature: Sampling temperature / サンプリング温度
        top_k: Top-K filtering / Top-Kフィルタリング
        top_p: Top-P (nucleus) filtering / Top-P（核）フィルタリング
        device: Device to use / 使用するデバイス
        verbose: Print generation steps / 生成ステップを表示

    Returns:
        Generated text / 生成されたテキスト
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if verbose:
        print("\n" + "=" * 60)
        print("Text Generation Process / テキスト生成プロセス")
        print("=" * 60)
        print(f"\nPrompt: '{prompt}'")
        print(f"プロンプト: '{prompt}'")
        print(f"\nSettings / 設定:")
        print(f"  Temperature: {temperature}")
        print(f"  Top-K: {top_k}")
        print(f"  Top-P: {top_p}")
        print(f"  Max new tokens: {max_new_tokens}")
        print(f"  最大生成トークン数: {max_new_tokens}")

    # Step 1: Encode prompt
    # ステップ1：プロンプトをエンコード
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    if verbose:
        print(f"\nStep 1: Encode prompt / ステップ1：プロンプトをエンコード")
        print(f"  Token IDs: {prompt_tokens[:10]}..." if len(prompt_tokens) > 10 else f"  Token IDs: {prompt_tokens}")

    # Step 2: Autoregressive generation
    # ステップ2：自己回帰生成
    if verbose:
        print(f"\nStep 2: Generate tokens one by one / ステップ2：トークンを1つずつ生成")
        print("-" * 40)

    generated_tokens = []

    for i in range(max_new_tokens):
        # Truncate if exceeds max sequence length
        # 最大系列長を超えたら切り詰める
        if input_ids.size(1) > model.max_seq_len:
            context = input_ids[:, -model.max_seq_len:]
        else:
            context = input_ids

        # Forward pass
        # フォワードパス
        logits = model(context)

        # Get logits for last position
        # 最後の位置のロジットを取得
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply temperature
        # 温度を適用
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        # Top-Kフィルタリングを適用
        if top_k is not None:
            top_k_val = min(top_k, next_token_logits.size(-1))
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Apply top-p filtering
        # Top-Pフィルタリングを適用
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')

        # Convert to probabilities
        # 確率に変換
        probs = torch.softmax(next_token_logits, dim=-1)

        # Sample next token
        # 次のトークンをサンプリング
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        # シーケンスに追加
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_tokens.append(next_token.item())

        # Decode the token
        # トークンをデコード
        if verbose and i < 10:
            token_char = tokenizer.decode([next_token.item()])
            top_prob = probs.max().item()
            print(f"  Token {i+1}: '{token_char}' (prob: {top_prob:.3f})")

    if verbose:
        if max_new_tokens > 10:
            print(f"  ... ({max_new_tokens - 10} more tokens)")
            print(f"  ... (残り {max_new_tokens - 10} トークン)")

    # Step 3: Decode output
    # ステップ3：出力をデコード
    full_output = tokenizer.decode(input_ids[0].tolist())

    if verbose:
        print("\n" + "-" * 40)
        print(f"Step 3: Decode output / ステップ3：出力をデコード")
        print(f"\nGenerated text / 生成されたテキスト:")
        print(f"'{full_output}'")

    return full_output


def interactive_mode(model: TinyLLM, tokenizer: SimpleTokenizer, device: torch.device):
    """
    Interactive text generation mode.
    対話型テキスト生成モード。

    Allows users to input prompts and see generated text.
    ユーザーがプロンプトを入力して生成されたテキストを見ることができます。
    """
    print("\n" + "=" * 60)
    print("Interactive Mode / 対話モード")
    print("=" * 60)
    print("Enter a prompt to generate text. Type 'quit' to exit.")
    print("プロンプトを入力してテキストを生成します。'quit'で終了。")
    print("-" * 60)

    while True:
        try:
            prompt = input("\nPrompt / プロンプト: ").strip()

            if prompt.lower() == 'quit':
                print("Goodbye! / さようなら！")
                break

            if not prompt:
                print("Please enter a prompt. / プロンプトを入力してください。")
                continue

            # Generate text
            # テキストを生成
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                device=device,
                verbose=True,
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! / 中断しました。さようなら！")
            break


def demonstrate_sampling_methods(
    model: TinyLLM,
    tokenizer: SimpleTokenizer,
    device: torch.device,
):
    """
    Demonstrate different sampling methods.
    異なるサンプリング手法のデモンストレーション。
    """
    print("\n" + "=" * 60)
    print("Sampling Methods Comparison / サンプリング手法の比較")
    print("=" * 60)

    prompt = "The "

    methods = [
        {"name": "Greedy (temp=0.1)", "temperature": 0.1, "top_k": None, "top_p": None},
        {"name": "Temperature=0.5", "temperature": 0.5, "top_k": None, "top_p": None},
        {"name": "Temperature=1.0", "temperature": 1.0, "top_k": None, "top_p": None},
        {"name": "Temperature=1.5", "temperature": 1.5, "top_k": None, "top_p": None},
        {"name": "Top-K=10", "temperature": 1.0, "top_k": 10, "top_p": None},
        {"name": "Top-P=0.9", "temperature": 1.0, "top_k": None, "top_p": 0.9},
        {"name": "Combined (temp=0.8, top_k=50)", "temperature": 0.8, "top_k": 50, "top_p": None},
    ]

    print(f"\nPrompt: '{prompt}'")
    print("-" * 60)

    for method in methods:
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=method["temperature"],
            top_k=method["top_k"],
            top_p=method["top_p"],
            device=device,
            verbose=False,
        )

        print(f"\n{method['name']}:")
        print(f"  '{generated[:80]}...'")


def main():
    """
    Main function for inference.
    推論のためのメイン関数。
    """
    parser = argparse.ArgumentParser(
        description="TinyLLM Inference / TinyLLM推論"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="tiny_llm_model.pt",
        help="Path to trained model / 学習済みモデルへのパス"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Prompt for generation / 生成用プロンプト"
    )
    parser.add_argument(
        "--max-tokens", "-n",
        type=int,
        default=100,
        help="Maximum tokens to generate / 生成する最大トークン数"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.8,
        help="Sampling temperature / サンプリング温度"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Top-K sampling / Top-Kサンプリング"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-P (nucleus) sampling / Top-P（核）サンプリング"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode / 対話モード"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demonstrate sampling methods / サンプリング手法のデモ"
    )

    args = parser.parse_args()

    # Check if model exists
    # モデルが存在するか確認
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"エラー: モデルファイルが見つかりません: {args.model}")
        print("\nPlease train a model first using train.py")
        print("まずtrain.pyを使用してモデルを学習してください")
        return

    # Device selection
    # デバイス選択
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"使用デバイス: {device}")

    # Load model
    # モデルを読み込む
    model, tokenizer = load_model(args.model, device)

    if args.demo:
        demonstrate_sampling_methods(model, tokenizer, device)
    elif args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.prompt:
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            verbose=True,
        )
    else:
        print("\nPlease provide a prompt with --prompt or use --interactive mode")
        print("--promptでプロンプトを指定するか、--interactiveモードを使用してください")


# Example usage without loading a trained model
# 学習済みモデルを読み込まない使用例
if __name__ == "__main__":
    # Check if we have a trained model
    # 学習済みモデルがあるか確認
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        print("=" * 60)
        print("TinyLLM Inference Demo / TinyLLM推論デモ")
        print("=" * 60)
        print("\nThis script can be run in several ways:")
        print("このスクリプトは以下の方法で実行できます：")
        print("\n1. With a trained model / 学習済みモデルを使用:")
        print("   python inference.py --model tiny_llm_model.pt --prompt 'Hello'")
        print("\n2. Interactive mode / 対話モード:")
        print("   python inference.py --model tiny_llm_model.pt --interactive")
        print("\n3. Sampling methods demo / サンプリング手法デモ:")
        print("   python inference.py --model tiny_llm_model.pt --demo")

        print("\n" + "-" * 60)
        print("Quick Demo (without trained model) / クイックデモ（学習済みモデルなし）")
        print("-" * 60)

        # Create a small model for demo
        # デモ用に小さなモデルを作成
        from tiny_transformer import TinyLLM

        tokenizer = SimpleTokenizer()
        sample_text = "Hello world! This is a demo."
        tokenizer.build_vocab([sample_text])

        model = TinyLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=128,
            max_seq_len=64,
        )

        device = torch.device('cpu')

        print("\nGenerating with untrained model (random output expected):")
        print("未学習モデルで生成（ランダムな出力が期待されます）：")

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt="Hello",
            max_new_tokens=20,
            temperature=1.0,
            device=device,
            verbose=True,
        )

        print("\nNote: The output is random because the model is not trained.")
        print("注意：モデルが学習されていないため、出力はランダムです。")
        print("Train the model first with train.py for meaningful output.")
        print("意味のある出力を得るには、まずtrain.pyでモデルを学習してください。")
