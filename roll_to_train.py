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
        self.accumulation_steps = 64  # Number of accumulation steps
        self.intelligence = intelligence
        self.dc = dc
        self.intelligence_modifier = (intelligence - 10) // 2
        self._loss_history = []
        self._modified_loss_history = []
        self._eval_loss_history = []
        self._grad_accum_counter = 0  # Track accumulated gradients
        self._accumulated_loss = 0


    @staticmethod
    def roll_d20():
        return torch.randint(1, 21, (1,)).item()

    def weight_update(self, loss, step, steps):
        roll = self.roll_d20()
        modified_roll = roll + self.intelligence_modifier
        print(f"Rolled a {roll} (Modified: {modified_roll})")
        self._accumulated_loss += loss.item()

        # Critical Fail: Roll a natural 1
        if roll == 1:
            print("Critical Fail! Large scaled loss applied!")
            loss = loss * 5.0
            self._modified_loss_history.append(loss.item())
            loss.backward()

        # Critical Success: Roll a natural 20
        elif roll == 20:
            print("Critical Success! Applying full loss.")
            loss.backward()
            self._modified_loss_history.append(loss.item())

        # Success: Roll > DC
        elif modified_roll > self.dc:
            scale = 1 / (modified_roll - self.dc)  # Example scaling
            print(f"Success! Scaling loss by {scale:.2f}.")
            loss = loss * scale
            self._modified_loss_history.append(loss.item())
            loss.backward()

        # Fail: Roll <= DC
        else:
            print("Fail! No update applied")
            self._modified_loss_history.append(0.0)
            self._grad_accum_counter += 1  # Still count for accumulation
            return

        # Increment the gradient accumulation counter
        self._grad_accum_counter += 1

        # Perform optimizer step and clear gradients after `accumulation_steps`
        if (self._grad_accum_counter >= self.accumulation_steps) or (step == steps-1):
            print("Performing optimizer step after gradient accumulation")

            self._loss_history.append(self._accumulated_loss / self.accumulation_steps)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Optional grad clipping
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulated_loss = 0
            self._grad_accum_counter = 0  # Reset the counter

    def step_lr(self):
        self.lr_scheduler.step()

    # Example training loop
    def train(self, dataloader, eval_dataloader, steps=3, eval_steps=100):

        step = 0
        self.model.train()
        while step < steps:
            self.model.train()
            for batch in dataloader:
                print(f"Step {step + 1}")
                inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
                labels = torch.tensor(batch["label"]).to(device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

                self.weight_update(loss, step, steps)
                step += 1
                if step >= steps:
                    break

                if step  % eval_steps == 0:
                    self.model.eval()
                    total_loss = 0
                    with torch.no_grad():
                        for batch in eval_dataloader:
                            inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt",
                                                    max_length=512).to(device)
                            labels = torch.tensor(batch["label"]).to(device)

                            outputs = self.model(**inputs, labels=labels)
                            total_loss += outputs.loss.item()

                    avg_loss = total_loss / len(eval_dataloader)
                    self._eval_loss_history.append(avg_loss)
                    print(f"Evaluation Loss: {avg_loss:.4f}")
                self.step_lr()
        fig, axes = plt.subplots(3, 1, figsize=(20, 30), sharex=True)

        # Loss Before Roll
        accumulation_range = [i for i in range(self.accumulation_steps, steps-1, self.accumulation_steps)]
        axes[0].plot(accumulation_range, self._loss_history, color='blue', linestyle='--', marker='o')
        axes[0].set_title('Loss Before Roll')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Loss After Roll
        axes[1].plot(self._modified_loss_history, color='green', linestyle='-', marker='x')
        axes[1].set_title('Loss After Roll')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Eval Loss
        steps_range = [i for i in range(eval_steps, steps, eval_steps)]
        axes[2].plot(steps_range, self._eval_loss_history, color='red', linestyle='-', marker='s')
        axes[2].set_title('Evaluation Loss')
        axes[2].set_xlabel('Training Steps')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # Save the figure
        plt.tight_layout()
        plt.savefig("roll_to_train_loss_subplots.png")
        print("Saved loss plots as 'roll_to_train_loss_subplots.png'")

