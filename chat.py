#!/usr/bin/env python3
"""
Interactive chat script for talking with trained SFT or RFT models.
"""

import sys
from homework.sft import load as load_sft
from homework.rft import load as load_rft


def chat_with_model(model, model_name="Model"):
    """Interactive chat loop with a model."""
    print(f"\n{'='*60}")
    print(f"Chatting with {model_name}")
    print(f"{'='*60}")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print(f"{'='*60}\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'clear':
                print("\n[Conversation cleared]\n")
                continue
            
            print(f"\n{model_name}: ", end="", flush=True)
            answer = model.generate(model.format_prompt(question))
            print(answer)
            
            # Try to parse the answer
            parsed = model.parse_answer(answer)
            if parsed == parsed:  # Check if not NaN
                print(f"\n[Parsed answer: {parsed}]\n")
            else:
                print(f"\n[Could not parse answer]\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main function to select and chat with a model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive chat with trained SFT or RFT models")
    parser.add_argument("model", nargs="?", choices=["sft", "rft"], help="Model to use (sft or rft)")
    parser.add_argument("--sft", action="store_true", help="Use SFT model")
    parser.add_argument("--rft", action="store_true", help="Use RFT model")
    
    args = parser.parse_args()
    
    # Determine which model to use
    if args.sft:
        model_type = 'sft'
    elif args.rft:
        model_type = 'rft'
    elif args.model:
        model_type = args.model.lower()
    else:
        print("Available models:")
        print("  1. SFT (Supervised Fine-Tuning)")
        print("  2. RFT (Rejection Sampling Fine-Tuning)")
        print()
        choice = input("Select model (1 or 2, or 'sft'/'rft'): ").strip().lower()
        
        if choice in ['1', 'sft']:
            model_type = 'sft'
        elif choice in ['2', 'rft']:
            model_type = 'rft'
        else:
            print("Invalid choice. Defaulting to SFT.")
            model_type = 'sft'
    
    print("\nLoading model...")
    try:
        if model_type == 'sft':
            model = load_sft()
            model_name = "SFT Model"
        elif model_type == 'rft':
            model = load_rft()
            model_name = "RFT Model"
        else:
            print(f"Unknown model type: {model_type}")
            print("Available: 'sft' or 'rft'")
            return
        
        chat_with_model(model, model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

