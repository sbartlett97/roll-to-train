import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from roll_to_train import RollToTrain


def main(intelligence=15, dc=12.0, dataset=None):
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    val_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = RollToTrain(model, tokenizer, optimizer, scheduler, intelligence, float(dc))
    trainer.train(dataloader, val_dataloader)
    trainer = RollToTrain(model, tokenizer, optimizer, scheduler, intelligence, float(dc),
                         mode="per_accumulation_step")
    trainer.train(dataloader, val_dataloader)


if __name__=="__main__":
    main()
