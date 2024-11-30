import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from roll_to_train import DnDTrainer
from datasets import load_dataset

def main(intelligence=15, dc=12, dataset=None):
    # Model and dataset setup (example)
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", streaming=True, split="train")

    dataloader = DataLoader(dataset, batch_size=4)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = DnDTrainer(model, tokenizer, optimizer, scheduler, intelligence, dc)
    trainer.train(dataloader, steps=1000)




if __name__=="__main__":
    main()