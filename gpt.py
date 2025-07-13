import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
import argparse
from datetime import datetime
import os
from tqdm import tqdm
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a GPT model on TinyStories dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of sequences processed in parallel",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=64,
        help="Maximum context length for predictions",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=2000, help="Maximum training iterations"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Evaluation interval in iterations",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--eval-iterations",
        type=int,
        default=100,
        help="Number of evaluation iterations",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=384, help="Embedding dimension size"
    )
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.2, help="Dropout probability"
    )
    parser.add_argument(
        "--num-stories",
        type=int,
        default=12000,
        help="Number of stories to load from dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model and outputs",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tinystories_bpe_tokenizer/4K",
        help='Tokenizer name or path (e.g., "gpt2" or "tinystories_bpe_tokenizer/1K")',
    )
    parser.add_argument('--use-torch-compile', action='store_true', help='Use torch.compile for model optimization')
    return parser.parse_args()


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionHead(nn.Module):
    def __init__(self, head_size, context_length, embedding_dim, dropout_rate):
        super().__init__()
        self.key_layer = nn.Linear(embedding_dim, head_size, bias=False)
        self.query_layer = nn.Linear(embedding_dim, head_size, bias=False)
        self.value_layer = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        keys = self.key_layer(inputs)  # (batch_size, seq_len, head_size)
        queries = self.query_layer(inputs)  # (batch_size, seq_len, head_size)
        values = self.value_layer(inputs)  # (batch_size, seq_len, head_size)

        if False:
            attention_scores = (
                queries @ keys.transpose(-2, -1) * keys.shape[-1] ** -0.5
            )  # (batch_size, seq_len, seq_len)
            attention_scores = attention_scores.masked_fill(
                self.tril_mask[:seq_len, :seq_len] == 0, float("-inf")
            )
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            return attention_weights @ values  # (batch_size, seq_len, head_size)
        else:
            return torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.dropout.p if self.dropout.training else 0.0,
                is_causal=True,
            )


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads, head_size, context_length, embedding_dim, dropout_rate
    ):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(head_size, context_length, embedding_dim, dropout_rate)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = torch.cat([head(inputs) for head in self.attention_heads], dim=-1)
        return self.dropout(self.projection(outputs))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, inputs):
        return self.network(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, dropout_rate):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.multi_head_attention = MultiHeadAttention(
            num_heads, head_size, context_length, embedding_dim, dropout_rate
        )
        self.feed_forward = FeedForward(embedding_dim, dropout_rate)
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)

    def forward(self, inputs):
        x = inputs + self.multi_head_attention(self.norm1(inputs))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        context_length,
        num_layers,
        num_heads,
        dropout_rate,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, context_length, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(
            torch.arange(seq_len, device=input_ids.device)
        )
        x = token_embeds + position_embeds
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.output_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, input_ids, max_new_tokens, context_length):
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids[:, -context_length:]
            logits, _ = self(input_ids_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids


def get_batch(data, batch_size, context_length, device):
    indices = torch.randint(len(data) - context_length, (batch_size,))
    inputs = torch.stack([data[i : i + context_length] for i in indices])
    targets = torch.stack([data[i + 1 : i + context_length + 1] for i in indices])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model, train_data, val_data, eval_iterations, batch_size, context_length, device
):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            with torch.amp.autocast('cuda'):
                _, loss = model(inputs, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    args = parse_arguments()
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('high')

    # Save model with timestamp and tokenizer info
    tokenizer_name = args.tokenizer.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = f"model_{args.embedding_dim}d_{args.num_layers}l_{args.num_heads}h_{tokenizer_name}_{timestamp}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    try:
        if os.path.exists(args.tokenizer):
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"Loaded local tokenizer from {args.tokenizer}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"Loaded tokenizer {args.tokenizer} from HuggingFace")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    text = "\n".join(dataset["train"]["text"][: args.num_stories])

    # Prepare data
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_split = int(0.9 * len(data))
    train_data = data[:train_split]
    val_data = data[train_split:]

    print(
        f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}"
    )

    # Initialize model
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    ).to(device)
    
    if args.use_torch_compile:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Error compiling model with torch.compile: {e}")
            print("Continuing without torch.compile...")

    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(p.numel() for p in model.token_embedding.parameters())
    print(
        f"Model parameters: {total_params / 1e6:.2f}M ({embedding_params / 1e6:.2f}M embedding)"
    )
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=200,
        num_training_steps=args.max_iterations
    )

    # Training loop
    for iteration in tqdm(range(args.max_iterations)):
        if iteration % args.eval_interval == 0 or iteration == args.max_iterations - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.eval_iterations,
                args.batch_size,
                args.context_length,
                device,
            )
            # print(
            #     f"Iteration {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            # )
            tqdm.write(
                f"Iteration {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            with open(
                os.path.join(args.output_dir, f"{model_name}_losses.txt"), "a"
            ) as f:
                f.write(f"{iteration}\t{losses['train']:.4f}\t{losses['val']:.4f}\n")

        inputs, targets = get_batch(
            train_data, args.batch_size, args.context_length, device
        )
        logits, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model_path = os.path.join(args.output_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Generate sample output
    context = torch.tensor(
        tokenizer.encode("Once\n"), dtype=torch.long, device=device
    ).unsqueeze(0)
    generated_tokens = model.generate(
        context, max_new_tokens=500, context_length=args.context_length
    )[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    print("\nGenerated sample:")
    print(generated_text)

    # Save generated text
    generated_text_path = os.path.join(args.output_dir, f"{model_name}_generated.txt")
    with open(generated_text_path, "w") as f:
        f.write(generated_text)
    print(f"Generated text saved to {generated_text_path}")


if __name__ == "__main__":
    main()
