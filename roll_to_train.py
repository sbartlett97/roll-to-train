"""roll-to-train is an experimental machine learning training
approach that introduces randomness to the training process on weight updates
using dice-roll mechanics from TTRPGs like Dungeons & Dragons."""

import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# D&D Trainer Class
class RollToTrain:
    """Main trainer class"""
    def __init__(self, model, tokenizer, optimizer, lr_scheduler, intelligence=10, dc=15,
                 accumulation_steps=64, mode="per_mini_batch", num_epochs=3):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = GradScaler()  # For mixed precision training

        # Hyperparameters and attributes
        self.accumulation_steps = accumulation_steps
        self.intelligence = intelligence
        self.dc = dc
        self.intelligence_modifier = (intelligence - 10) // 2
        self._loss_history = []
        self._modified_loss_history = []
        self._eval_loss_history = []
        self._grad_accum_counter = 0
        self._accumulated_loss = 0
        self._mode = mode
        self.epoch = 0
        self.epochs = num_epochs

    def roll_d20(self):
        """Roll a D20 dice on the GPU."""
        return torch.randint(1, 21, (1,), device=device).item() + self.intelligence_modifier

    def get_scale_factor(self, roll):
        """Compute the scaling factor based on the roll result."""
        scale = torch.zeros(1, device=device)
        if roll == 1:
            print("Critical Fail! Inverse loss applied!")
            scale = -1.0
        elif roll == 20:
            print("Critical Success! Applying full loss.")
            scale = 1.0
        elif roll >= self.dc:
            scale = 1 * ((roll - self.dc) / 100)
            print(f"Success! Scaling loss by {scale:.2f}.")
        else:
            scale = 0.0
            print("Fail! No update applied")
        return scale

    def weight_update(self, loss):
        """Accumulate gradients and perform an update after the accumulation steps."""
        self._grad_accum_counter += 1
        roll = self.roll_d20()
        scale = self.get_scale_factor(roll)

        with autocast():
            if self._mode == 'per_mini_batch':
                loss = loss * scale
                self._modified_loss_history.append(loss.item())
            self.scaler.scale(loss).backward()

        self._accumulated_loss += loss.item()

        if self._grad_accum_counter >= self.accumulation_steps:
            print("Performing optimizer step after gradient accumulation")
            self._loss_history.append(self._accumulated_loss / self.accumulation_steps)
            if self._mode == 'per_accumulation_step':
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale)

                self._modified_loss_history.append((self._accumulated_loss / self.accumulation_steps)*scale)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            self._accumulated_loss = 0
            self._grad_accum_counter = 0

    def train(self, train_dataloader, eval_dataloader):
        """Train the model for a specified number of steps."""
        self.epoch = 0

        while self.epoch < self.epochs:
            for batch_idx, batch in enumerate(train_dataloader):
                if self.model.eval:
                    self.model.train()

                print(f"Step {self.epoch + 1}, Batch {batch_idx + 1}")
                inputs = self.tokenizer(batch["text"], padding=True, truncation=True,
                                        return_tensors="pt", max_length=512).to(device)
                labels = batch["label"].to(device)

                with autocast():
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                self.weight_update(loss)

                if self.epoch >= self.epochs:
                    break

            self.evaluate(eval_dataloader)
            self.lr_scheduler.step()
            self.epoch += 1
        self.plot_loss(len(train_dataloader))

    def evaluate(self, eval_dataloader):
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for val_batch in eval_dataloader:
                inputs = self.tokenizer(val_batch["text"], padding=True, truncation=True,
                                        return_tensors="pt", max_length=512).to(device)
                labels = val_batch["label"].to(device)

                outputs = self.model(**inputs, labels=labels)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(eval_dataloader)
        self._eval_loss_history.append(avg_loss)
        print(f"Evaluation Loss: {avg_loss:.4f}")

    def plot_loss(self, steps):
        """Plot and save the training and evaluation loss."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 20), sharex=True)

        # Loss Before Roll
        axes[0].plot(self._loss_history, color='blue', linestyle='--', marker='o')
        axes[0].set_title('Loss Before Roll')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Loss After Roll
        modified_loss_steps = [i for i in range(steps*self.epochs)] if self._mode == "per_mini_batch" else [i for i in range(0, steps*self.epochs,
                                                                                                        self.accumulation_steps)]
        axes[1].plot(modified_loss_steps, self._modified_loss_history, color='green', linestyle='-', marker='x')
        axes[1].set_title('Loss After Roll')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Evaluation Loss
        eval_steps = [i for i in range(0, steps*self.epochs, steps)]
        axes[2].plot(eval_steps, self._eval_loss_history, color='red', linestyle='-', marker='s')
        axes[2].set_title('Evaluation Loss')
        axes[2].set_xlabel('Training Steps')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{self._mode}_roll_to_train_loss_subplots.png")
        print(f"Saved loss plots as '{self._mode}_roll_to_train_loss_subplots.png'")
