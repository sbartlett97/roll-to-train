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
        self._modified_loss_history = []
        self._eval_loss_history = []

    @staticmethod
    def roll_d20():
        return torch.randint(1, 21, (1,)).item()

    def weight_update(self, loss):
        roll = self.roll_d20()
        modified_roll = roll + self.intelligence_modifier
        print(f"Rolled a {roll} (Modified: {modified_roll})")
        self._loss_history.append(loss.item())
        # Critical Fail: Roll a natural 1
        if roll == 1:
            print("Critical Fail! Large scaled loss applied!")
            loss = loss * 5.0
            self._modified_loss_history.append(loss.item())
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Optional grad clipping
            self.optimizer.step()

        # Critical Success: Roll a natural 20
        elif roll == 20:
            print("Critical Success! Applying full loss.")
            loss.backward()
            self._modified_loss_history.append(loss.item())
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Optional grad clipping
            self.optimizer.step()

        # Success: Roll > DC
        elif modified_roll > self.dc:
            scale = 1 / (modified_roll - self.dc)  # Example scaling
            print(f"Success! Scaling loss by {scale:.2f}.")
            loss = loss * scale
            self._modified_loss_history.append(loss.item())
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Fail: Roll <= DC
        else:
            print("Fail! No update applied")
            self._modified_loss_history.append(0.0)
            self.optimizer.zero_grad()
            return

        # Clear gradients after update
        self.optimizer.zero_grad()

    def step_lr(self):
        self.lr_scheduler.step()

    # Example training loop
    def train(self, dataloader, eval_dataloader, steps=3, eval_steps=100):

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
            if step  % eval_steps == 0:
                self.model.eval()
                total_loss = 0
                with torch.no_grad():
                    for batch in eval_dataloader:
                        inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt",
                                                max_length=256).to(device)
                        labels = torch.tensor(batch["label"]).to(device)

                        outputs = self.model(**inputs, labels=labels)
                        total_loss += outputs.loss.item()

                avg_loss = total_loss / len(eval_dataloader)
                self._eval_loss_history.append(avg_loss)
                print(f"Evaluation Loss: {avg_loss:.4f}")
        plt.figure(figsize=(10, 6))
        plt.plot(self._loss_history, label='Loss Before Roll', color='blue', linestyle='--', marker='o')
        plt.plot(self._modified_loss_history, label='Loss After Roll', color='green', linestyle='-', marker='x')
        plt.plot(self._eval_loss_history, label='Validation Loss', color='red', linestyle='.', marker='x')
        # Add labels, title, and legend
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss History with Gamified Updates')
        plt.legend()

        # Show grid and plot
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("roll_to_train_loss.png")

