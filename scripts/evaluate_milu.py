import os
import argparse
import random
import re
import json
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from tqdm import tqdm
from collections import defaultdict

# Setup logger
logger = logging.getLogger(__name__)


class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequence_ids):
        super().__init__()
        self.stop_sequence_ids = stop_sequence_ids
        self.match_len = len(stop_sequence_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.match_len:
            return False
        if (
            list(input_ids[0][-self.match_len :].cpu().numpy())
            == self.stop_sequence_ids
        ):
            return True
        return False


def extract_think_and_response(text):
    """
    Extracts <think> and <response> content from a given text.

    Returns a tuple: (think_content, response_content)
    """
    # Use non-greedy matching and allow multiline
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    response_match = re.search(
        r"<response>(.*?)</response>", text, re.DOTALL | re.IGNORECASE
    )

    think = think_match.group(1).strip() if think_match else ""
    response = response_match.group(1).strip() if response_match else ""

    return think, response


def save_to_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MILU dataset")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/dccstor/indiclm/rudra/granite-3.3-8b-instruct/r250409a/",
        help="Hugging Face model",
    )
    parser.add_argument(
        "--samples", type=int, default=-1, help="Samples per group (-1 = all)"
    )
    parser.add_argument("--shots", type=int, default=0, help="Few-shot count")
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Tokens to generate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--language", type=str, default=None, help="Specify the language for evaluation"
    )

    parser.add_argument("--output_file", type=str, help="Path to save output file")
    return parser.parse_args()


def build_chat(example, train_set=None, shots=0):
    messages = []

    if shots > 0 and train_set:
        few_shots = random.sample(train_set, shots)
        for shot in few_shots:
            choices = f"A. {shot['option1']}\nB. {shot['option2']}\nC. {shot['option3']}\nD. {shot['option4']}\n"

            question = f"Question:\n{shot['question']}\nChoices:\n{choices}"
            messages.append({"role": "user", "content": question})
            if shot["target"] == "option1":
                messages.append({"role": "assistant", "content": "Answer: A\n"})
            elif shot["target"] == "option2":
                messages.append({"role": "assistant", "content": "Answer: B\n"})
            elif shot["target"] == "option3":
                messages.append({"role": "assistant", "content": "Answer: C\n"})
            elif shot["target"] == "option4":
                messages.append({"role": "assistant", "content": "Answer: D\n"})
            else:
                raise ValueError(f"Wrong correct answer in {shot}")

    choices = f"A. {example['option1']}\nB. {example['option2']}\nC. {example['option3']}\nD. {example['option4']}\n"

    question = f"Question:\n{example['question']}\nChoices:\n{choices}"

    messages.append({"role": "user", "content": question})
    return messages


def evaluate(args):
    random.seed(args.seed)

    generations = []

    # Load datasets
    test_set = load_dataset(
        "murthyrudra/milu-cleaned", data_dir=args.language, split="test", token=True
    )
    train_set = (
        load_dataset(
            "murthyrudra/milu-cleaned",
            data_dir=args.language,
            split="validation",
            token=True,
        )
        if args.shots > 0
        else None
    )

    # Load model + tokenizer
    logger.info(f"🔄 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.eval()

    # Encode stop sequence
    stop_ids = tokenizer.encode("</response>", add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([StopOnSequence(stop_ids)])

    # Group by domain
    grouped = defaultdict(list)
    for ex in test_set:
        grouped[ex["domain"]].append(ex)

    # Evaluate groups
    report = {}
    for domain, examples in grouped.items():
        subset = (
            random.sample(examples, args.samples)
            if args.samples > 0 and len(examples) > args.samples
            else examples
        )

        logger.info(f"\n🌍 Evaluating {len(subset)} in ({args.language}, {domain})")

        for start_idx in tqdm(range(0, len(subset), args.batch_size)):
            batch = subset[start_idx : start_idx + args.batch_size]
            chats = [build_chat(ex, train_set, args.shots) for ex in batch]
            prompts = [
                tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                for chat in chats
            ]

            input_ids = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            for i, ex in enumerate(batch):
                generated = tokenizer.decode(
                    outputs[i][input_ids.shape[0] :], skip_special_tokens=True
                )

                ex["thinking"], ex["response"] = extract_think_and_response(generated)
                generations.append(ex)

    save_to_jsonl(generations, os.path.join(args.output_file, f"{args.language}.jsonl"))


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
