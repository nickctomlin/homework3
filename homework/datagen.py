def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate RFT dataset by:
    1. Using CoTModel to generate multiple completions for each question
    2. Selecting the one with the correct answer
    3. Saving to JSON file
    """
    import json
    from pathlib import Path
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    # Convert parameters to correct types (Fire may pass them as strings)
    oversample = int(oversample)
    temperature = float(temperature)
    
    # Use the instruct model for better CoT reasoning
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    train_dataset = Dataset("train")
    
    rft_data = []
    
    print(f"Generating RFT dataset for {len(train_dataset)} questions...")
    for idx, (question, correct_answer) in enumerate(train_dataset):
        if idx % 10 == 0:
            print(f"Processing question {idx}/{len(train_dataset)}")
        
        # Generate multiple completions
        generations = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # generations is a list of lists, get the first (and only) prompt's generations
        completions = generations[0] if isinstance(generations[0], list) else [generations[0]]
        
        # Find the first completion with a correct answer
        found_correct = False
        for completion in completions:
            parsed_answer = model.parse_answer(completion)
            # Check if not NaN (NaN != NaN is True, so not NaN means parsed_answer == parsed_answer)
            if parsed_answer == parsed_answer:  # Check if not NaN
                if is_answer_valid(parsed_answer, correct_answer):
                    # Found a correct answer, add to dataset
                    rft_data.append([question, float(correct_answer), completion])
                    found_correct = True
                    break
        
        # If no correct answer found, skip this data point (as per instructions)
        if not found_correct:
            continue
    
    # Save to JSON file
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"Generated {len(rft_data)} valid examples out of {len(train_dataset)} questions")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
