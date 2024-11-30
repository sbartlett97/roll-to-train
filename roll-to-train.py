import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

# D&D Trainer class
class DnDTrainer:
    def __init__(self, model, optimizer, lr_scheduler, intelligence=10, dc=15):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.intelligence = intelligence
        self.dc = dc
        self.intelligence_modifier = (intelligence - 10) // 2

    def roll_d20(self):
        return torch.randint(1, 21, (1,)).item()

    def weight_update(self, loss):
        roll = self.roll_d20()
        modified_roll = roll + self.intelligence_modifier
        print(f"Rolled a {roll} (Modified: {modified_roll})")

        # Critical Fail: Roll a natural 1
        if roll == 1:
            print("Critical Fail! Loss is zeroed out.")
            return  # Skip weight updates

        # Critical Success: Roll a natural 20
        elif roll == 20:
            print("Critical Success! Applying full loss.")
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Optional grad clipping
            self.optimizer.step()

        # Success: Roll > DC
        elif modified_roll > self.dc:
            scale = (modified_roll - self.dc) / 5.0  # Example scaling
            print(f"Success! Scaling loss by {scale:.2f}.")
            (loss * scale).backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Fail: Roll <= DC
        else:
            print("Fail! Applying small inverse weight update.")
            (-0.01 * loss).backward()  # Small negative weight update
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Clear gradients after update
        self.optimizer.zero_grad()

    def step_lr(self):
        self.lr_scheduler.step()

# Example training loop
def train(model, tokenizer, dataset, epochs=3, batch_size=16, intelligence=10, dc=15):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = DnDTrainer(model, optimizer, scheduler, intelligence, dc)

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for batch in dataloader:
            inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
            labels = torch.tensor(batch["labels"])

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            trainer.weight_update(loss)

        trainer.step_lr()

# Model and dataset setup (example)
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example dataset (you would replace this with your actual dataset)
dataset = [{"text": "Example text.", "labels": 1} for _ in range(100)]

train(model, tokenizer, dataset)
