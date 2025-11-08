import json
import torch
from pathlib import Path

from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset


class RFTModel(BaseLLM):
    """RFT model that formats prompts correctly for inference"""
    def format_prompt(self, question: str) -> str:
        """
        Format prompt to match training format: question + space
        During training, format was: question + " " + reasoning (which includes answer)
        """
        return f"{question} "


def load() -> RFTModel:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_rft_example(question: str, answer: str, reasoning: str) -> dict[str, str]:
    """
    Format RFT example where reasoning already contains the answer in <answer> tags.
    We train on question + reasoning (the full chain of thought).
    """
    return {
        "question": question,
        "answer": reasoning
    }


class RFTDataset:
    """Dataset for RFT training data loaded from JSON"""
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_model(
    output_dir: str = None,
    **kwargs,
):
    """
    Train RFT model on question + reasoning pairs.
    Reuses much of the SFT code but trains on RFT dataset format.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "rft_model")
    
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(f"RFT dataset not found at {rft_data_path}. Run datagen.py first.")
    
    rft_dataset = RFTDataset(str(rft_data_path))
    
    llm = BaseLLM()
    
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=16,
        lora_alpha=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    llm.model = get_peft_model(llm.model, lora_config)
    
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()
    
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_dataset, format_rft_example)
    
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
    
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model()
    
    rft_model_path = Path(__file__).parent / "rft_model"
    trainer.save_model(str(rft_model_path))
    
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
