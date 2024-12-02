import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from roll_to_train import DnDTrainer
from datasets import load_dataset

def main(intelligence=15, dc=12, dataset=None):
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    val_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = DnDTrainer(model, tokenizer, optimizer, scheduler, intelligence, dc)
    trainer.train(dataloader, val_dataloader, steps=len(dataloader), eval_steps=100)




if __name__=="__main__":
    main()