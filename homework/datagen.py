def generate_dataset(output_json="data/rft.json", oversample=10, temperature=0.6):
    """
    Generate RFT dataset by:
    1. Using CoTModel to generate multiple completions for each question
    2. Selecting the one with the correct answer
    3. Saving to JSON file
    
    Args:
        output_json: Path to output JSON file (default: "data/rft.json")
        oversample: Number of generations per question (default: 10)
        temperature: Sampling temperature (default: 0.6)
    """
    import json
    from pathlib import Path
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid
    
    # Convert parameters to correct types (Fire passes everything as strings from command line)
    # Safe conversion that handles strings
    try:
        output_json = str(output_json)
    except:
        pass
    
    try:
        oversample = int(float(str(oversample)))
    except (ValueError, TypeError):
        oversample = 10  # default
    
    try:
        temperature = float(str(temperature))
    except (ValueError, TypeError):
        temperature = 0.6  # default
    
    # Use the instruct model for better CoT reasoning
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    train_dataset = Dataset("train")
    
    rft_data = []
    
    print(f"Generating RFT dataset for {len(train_dataset)} questions...")
    print(f"Using oversample={oversample}, temperature={temperature}")
    
    for idx, (question, correct_answer) in enumerate(train_dataset):
        if idx % 10 == 0:
            print(f"Processing question {idx}/{len(train_dataset)}")
        
        # Generate multiple completions
        # Need to format the prompt first (CoTModel.format_prompt handles the chat template)
        formatted_question = model.format_prompt(question)
        generations = model.batched_generate(
            [formatted_question],
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # generations is a list of lists, get the first (and only) prompt's generations
        completions = generations[0] if isinstance(generations[0], list) else [generations[0]]
        
        # Debug: print first few completions for first question
        if idx == 0 and len(completions) > 0:
            print(f"Sample completion: {completions[0][:200]}...")
        
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
                    if idx < 5:  # Debug first few successes
                        print(f"  ✓ Found correct answer: {parsed_answer} (expected: {correct_answer})")
                    break
        
        # If no correct answer found, skip this data point (as per instructions)
        if not found_correct and idx < 5:  # Debug first few failures
            print(f"  ✗ No correct answer found (got {len(completions)} completions)")
    
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
