"""roll-to-train is an experimental machine learning training
approach that introduces randomness to the training process on weight updates
using dice-roll mechanics from TTRPGs like Dungeons & Dragons."""

import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CharacterClass:
    """Base class for character classes"""
    def __init__(self, name, level=1):
        self.name = name
        self.level = level
        self.abilities = []

    def apply_ability(self, loss, roll):
        """Apply class-specific abilities to the training process"""
        return loss

class Wizard(CharacterClass):
    """Wizard class - specializes in learning rate manipulation"""
    def __init__(self, level=1):
        super().__init__("Wizard", level)
        self.abilities = ["Arcane Recovery", "Spell Mastery"]
        self.arcane_recovery_charges = 3

    def apply_ability(self, loss, roll):
        if roll >= 15 and self.arcane_recovery_charges > 0:
            print("Arcane Recovery! Adjusting learning rate...")
            self.arcane_recovery_charges -= 1
            return loss * 1.5  # Increase learning rate temporarily
        return loss

class Rogue(CharacterClass):
    """Rogue class - specializes in gradient manipulation"""
    def __init__(self, level=1):
        super().__init__("Rogue", level)
        self.abilities = ["Sneak Attack", "Cunning Action"]
        self.sneak_attack_dice = 1

    def apply_ability(self, loss, roll):
        if roll >= 12:
            print("Sneak Attack! Applying gradient boost...")
            return loss * (1 + 0.2 * self.sneak_attack_dice)
        return loss

class ExperienceSystem:
    """Manages experience points and leveling"""
    def __init__(self, base_xp=1000, xp_scaling=1.5):
        self.base_xp = base_xp
        self.xp_scaling = xp_scaling
        self.current_xp = 0
        self.level = 1
        self.xp_to_next_level = self.base_xp

    def add_xp(self, amount):
        """Add experience points and handle leveling up"""
        self.current_xp += amount
        while self.current_xp >= self.xp_to_next_level:
            self.level_up()

    def level_up(self):
        """Handle level up logic"""
        self.level += 1
        self.current_xp -= self.xp_to_next_level
        self.xp_to_next_level = int(self.base_xp * (self.xp_scaling ** (self.level - 1)))
        print(f"Level Up! Now level {self.level}")

    def get_level_bonus(self):
        """Get training bonus based on level"""
        return 0.1 * (self.level - 1)  # 10% bonus per level

# D&D Trainer Class
class RollToTrain:
    """Main trainer class"""
    def __init__(self, model, tokenizer, optimizer, lr_scheduler, intelligence=10, dc=15,
                 accumulation_steps=64, mode="per_mini_batch", num_epochs=3,
                 dice_type="d20", advantage=False, disadvantage=False,
                 character_class=None, use_xp_system=True):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a PyTorch module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a PyTorch optimizer")
        if not isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise TypeError("lr_scheduler must be a PyTorch learning rate scheduler")
        if not 1 <= intelligence <= 20:
            raise ValueError("intelligence must be between 1 and 20")
        if not 1 <= dc <= 30:
            raise ValueError("dc must be between 1 and 30")
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        if mode not in ["per_mini_batch", "per_accumulation_step"]:
            raise ValueError("mode must be either 'per_mini_batch' or 'per_accumulation_step'")
        if num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if dice_type not in ["d4", "d6", "d8", "d10", "d12", "d20", "d100"]:
            raise ValueError("dice_type must be one of: d4, d6, d8, d10, d12, d20, d100")
        if advantage and disadvantage:
            raise ValueError("Cannot have both advantage and disadvantage")

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
        self.dice_type = dice_type
        self.advantage = advantage
        self.disadvantage = disadvantage
        self.dice_sides = int(dice_type[1:])
        self.character_class = character_class
        self.use_xp_system = use_xp_system
        if use_xp_system:
            self.xp_system = ExperienceSystem()

    def roll_dice(self):
        """Roll dice with optional advantage/disadvantage."""
        if self.advantage:
            roll1 = torch.randint(1, self.dice_sides + 1, (1,), device=device).item()
            roll2 = torch.randint(1, self.dice_sides + 1, (1,), device=device).item()
            roll = max(roll1, roll2)
        elif self.disadvantage:
            roll1 = torch.randint(1, self.dice_sides + 1, (1,), device=device).item()
            roll2 = torch.randint(1, self.dice_sides + 1, (1,), device=device).item()
            roll = min(roll1, roll2)
        else:
            roll = torch.randint(1, self.dice_sides + 1, (1,), device=device).item()
        
        return roll + self.intelligence_modifier

    def get_scale_factor(self, roll):
        """Compute the scaling factor based on the roll result."""
        scale = torch.zeros(1, device=device)
        if roll == 1:
            print("Critical Fail! Inverse loss applied!")
            scale = -1.0
        elif roll == self.dice_sides:
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
        roll = self.roll_dice()
        scale = self.get_scale_factor(roll)

        with autocast():
            if self._mode == 'per_mini_batch':
                loss = loss * scale
                if self.character_class:
                    loss = self.character_class.apply_ability(loss, roll)
                if self.use_xp_system:
                    level_bonus = self.xp_system.get_level_bonus()
                    loss = loss * (1 + level_bonus)
                    # Award XP based on roll success
                    if roll >= self.dc:
                        self.xp_system.add_xp(int(roll * 10))
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
            print(f"\nEpoch {self.epoch + 1}/{self.epochs}")
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                if self.model.eval:
                    self.model.train()

                inputs = self.tokenizer(batch["text"], padding=True, truncation=True,
                                        return_tensors="pt", max_length=512).to(device)
                labels = batch["label"].to(device)

                with autocast():
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                self.weight_update(loss)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

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

    def save_checkpoint(self, path):
        """Save a checkpoint of the model and training state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': self.epoch,
            'loss_history': self._loss_history,
            'modified_loss_history': self._modified_loss_history,
            'eval_loss_history': self._eval_loss_history,
            'intelligence': self.intelligence,
            'dc': self.dc,
            'mode': self._mode,
            'dice_type': self.dice_type,
            'advantage': self.advantage,
            'disadvantage': self.disadvantage,
            'dice_sides': self.dice_sides,
            'character_class': self.character_class.__class__.__name__ if self.character_class else None,
            'use_xp_system': self.use_xp_system,
            'xp_system': self.xp_system.__dict__ if self.use_xp_system else None
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load a checkpoint of the model and training state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self._loss_history = checkpoint['loss_history']
        self._modified_loss_history = checkpoint['modified_loss_history']
        self._eval_loss_history = checkpoint['eval_loss_history']
        self.intelligence = checkpoint['intelligence']
        self.dc = checkpoint['dc']
        self._mode = checkpoint['mode']
        self.dice_type = checkpoint['dice_type']
        self.advantage = checkpoint['advantage']
        self.disadvantage = checkpoint['disadvantage']
        self.dice_sides = checkpoint['dice_sides']
        self.use_xp_system = checkpoint['use_xp_system']
        if self.use_xp_system:
            self.xp_system = ExperienceSystem()
            self.xp_system.__dict__.update(checkpoint['xp_system'])
        print(f"Loaded checkpoint from {path}")
