import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# D&D Trainer class
class DnDTrainer:
    def __init__(self, model, tokenizer, optimizer, lr_scheduler, intelligence=10, dc=15):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.intelligence = intelligence
        self.dc = dc
        self.intelligence_modifier = (intelligence - 10) // 2
        self._loss_history = []

    @staticmethod
    def roll_d20():
        return torch.randint(1, 21, (1,)).item()

    def weight_update(self, loss):
        roll = self.roll_d20()
        modified_roll = roll + self.intelligence_modifier
        print(f"Rolled a {roll} (Modified: {modified_roll})")

        # Critical Fail: Roll a natural 1
        if roll == 1:
            print("Critical Fail! Large inverse loss applied!")
            loss = loss * -5.0
            self._loss_history.append(loss.item())
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Optional grad clipping
            self.optimizer.step()

        # Critical Success: Roll a natural 20
        elif roll == 20:
            print("Critical Success! Applying full loss.")
            loss.backward()
            self._loss_history.append(loss.item())
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Optional grad clipping
            self.optimizer.step()

        # Success: Roll > DC
        elif modified_roll > self.dc:
            scale = (modified_roll - self.dc) / 5.0  # Example scaling
            print(f"Success! Scaling loss by {scale:.2f}.")
            loss = loss * scale
            self._loss_history.append(loss.item())
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Fail: Roll <= DC
        else:
            print("Fail! Applying small inverse weight update.")
            loss = -0.01 * loss
            self._loss_history.append(loss)
            loss.backward()  # Small negative weight update
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Clear gradients after update
        self.optimizer.zero_grad()

    def step_lr(self):
        self.lr_scheduler.step()

    # Example training loop
    def train(self, dataloader, steps=3):

        step = 0
        self.model.train()
        while step < steps:
            for batch in dataloader:
                print(f"Step {step + 1}")
                inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
                labels = torch.tensor(batch["label"]).to(device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

                self.weight_update(loss)
                step += 1
                if step >= steps:
                    break
            self.step_lr()
        plt.plot(self._loss_history)
        plt.savefig("roll_to_train_loss.png")

