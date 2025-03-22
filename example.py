import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from roll_to_train import RollToTrain, Wizard, Rogue
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

def prepare_dataset():
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Convert to PyTorch format
    tokenized_datasets.set_format("torch")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=8,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=8,
        shuffle=False
    )
    
    return train_dataloader, eval_dataloader, tokenizer

def main():
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    # Prepare dataset and tokenizer
    train_dataloader, eval_dataloader, tokenizer = prepare_dataset()
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = LinearLR(optimizer, step_size=1, gamma=0.1)  # Reduce LR by 0.1 each epoch
    
    # Initialize trainer with a Wizard character class
    trainer = RollToTrain(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        intelligence=15,  # High intelligence for better rolls
        dc=12,  # Lower DC for more frequent successes
        accumulation_steps=4,
        mode="per_mini_batch",
        num_epochs=3,
        dice_type="d20",
        character_class=Wizard(level=1),
        use_xp_system=True,
        encounter_chance=0.1
    )
    
    # Train the model
    trainer.train(train_dataloader, eval_dataloader)
    
    # Save checkpoint
    trainer.save_checkpoint("imdb_wizard_checkpoint.pt")

if __name__ == "__main__":
    main() 