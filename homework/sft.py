import torch

from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    """SFT model that formats prompts correctly for inference"""
    def format_prompt(self, question: str) -> str:
        """
        Format prompt using chat template to match training format.
        During training, we use chat template with question + answer.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a unit conversion assistant. You MUST end every answer with <answer>number</answer> where number is the final converted value."
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        formatted = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        return formatted


def load() -> SFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We use chat template format to match the base model's expected format.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a unit conversion assistant. You MUST end every answer with <answer>number</answer> where number is the final converted value."
        },
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]
    
    full_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False
    )
    
    full_text = full_text + tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    
    if hasattr(input_ids, 'tolist'):
        input_ids_list = input_ids.tolist()
    else:
        input_ids_list = list(input_ids)
    
    attention_mask = full["attention_mask"]
    if hasattr(attention_mask, "sum"):
        actual_length = int(attention_mask.sum().item())
    else:
        actual_length = int(sum(attention_mask))
    
    actual_tokens = input_ids_list[:actual_length]
    
    question_len = len(input_ids_list)
    answer_start_pattern = [11247, 46]
    
    pattern_found = False
    for i in range(len(actual_tokens) - len(answer_start_pattern) + 1):
        if actual_tokens[i:i+len(answer_start_pattern)] == answer_start_pattern:
            question_len = i
            pattern_found = True
            break
    
    if question_len >= len(actual_tokens):
        answer_start_text = "<answer>"
        answer_start_tokens = tokenizer(answer_start_text, add_special_tokens=False)["input_ids"]
        for i in range(len(actual_tokens) - len(answer_start_tokens) + 1):
            if actual_tokens[i:i+len(answer_start_tokens)] == answer_start_tokens:
                question_len = i
                break
    
    question_len = min(question_len, len(input_ids))

    if hasattr(input_ids, 'tolist'):
        input_ids_list_for_labels = input_ids.tolist()
    else:
        input_ids_list_for_labels = list(input_ids)
    
    labels = [-100] * question_len + input_ids_list_for_labels[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    rounded_answer = round(float(answer), 3)
    formatted_answer = f"<answer>{rounded_answer}</answer>"
    
    return {
        "question": prompt,
        "answer": formatted_answer
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = None,
    **kwargs,
):
    from pathlib import Path
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "sft_model")
    
    llm = BaseLLM()
    
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=8,
        lora_alpha=32,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    llm.model = get_peft_model(llm.model, lora_config)
    
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()
    
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=5e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_strategy="epoch",
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model()
    
    sft_model_path = Path(__file__).parent / "sft_model"
    trainer.save_model(str(sft_model_path))
    
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
