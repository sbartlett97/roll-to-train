import torch
import argparse
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from roll_to_train import RollToTrain

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using Roll-to-Train approach')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased",
                      help='Name of the pretrained model to use')
    parser.add_argument('--intelligence', type=int, default=15,
                      help='Intelligence score (1-20)')
    parser.add_argument('--dc', type=float, default=12.0,
                      help='Difficulty Class (1-30)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1,
                      help='Number of epochs to train')
    parser.add_argument('--accumulation_steps', type=int, default=64,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Load dataset
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        val_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
        
        # Create dataloaders
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Train with per_mini_batch mode
        print("\nTraining with per_mini_batch mode...")
        trainer = RollToTrain(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            intelligence=args.intelligence,
            dc=args.dc,
            accumulation_steps=args.accumulation_steps,
            mode="per_mini_batch",
            num_epochs=args.num_epochs
        )
        trainer.train(dataloader, val_dataloader)
        trainer.save_checkpoint(checkpoint_dir / "per_mini_batch_checkpoint.pt")
        
        # Train with per_accumulation_step mode
        print("\nTraining with per_accumulation_step mode...")
        trainer = RollToTrain(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            intelligence=args.intelligence,
            dc=args.dc,
            accumulation_steps=args.accumulation_steps,
            mode="per_accumulation_step",
            num_epochs=args.num_epochs
        )
        trainer.train(dataloader, val_dataloader)
        trainer.save_checkpoint(checkpoint_dir / "per_accumulation_step_checkpoint.pt")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
