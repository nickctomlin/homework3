import torch

from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Round the answer to a reasonable precision (3 decimal places)
    rounded_answer = round(float(answer), 3)
    
    # Format as question + answer with <answer> tags
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
    
    # Set default output directory
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "sft_model")
    
    # Initialize model and tokenizer
    llm = BaseLLM()
    
    # Configure LoRA
    # Using r=8 to keep model size below 20MB, lora_alpha=32 (4*8)
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=8,
        lora_alpha=32,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    llm.model = get_peft_model(llm.model, lora_config)
    
    # Enable input require grads if using GPU (for gradient checkpointing)
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()
    
    # Load and prepare dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=5e-4,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        save_strategy="epoch",
        logging_steps=10,
    )
    
    # Create trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save the model to the specified directory
    trainer.save_model()
    
    # Also save to the homework/sft_model directory as specified
    sft_model_path = Path(__file__).parent / "sft_model"
    trainer.save_model(str(sft_model_path))
    
    # Test the model
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
