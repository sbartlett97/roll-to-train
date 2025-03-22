# 🎲 Roll-to-Train: Where D&D Meets Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)](https://github.com/yourusername/roll-to-train)

> 🎯 **Transform your ML training into an epic adventure!** Roll-to-Train brings the excitement of tabletop RPGs to machine learning, making model training more engaging and potentially more robust.

## 🌟 Features

- 🎲 **D&D-Inspired Mechanics**: Roll for success, critical hits, and saving throws during training
- 🧙‍♂️ **Character Classes**: Choose your training style with Wizard, Rogue, and more
- 🐉 **Monster Encounters**: Face off against Gradient Ooze, Loss Dragon, and Weight Wraith
- 📈 **Experience System**: Level up your training process
- 🔄 **Flexible Integration**: Works with PyTorch and HuggingFace transformers
- 📊 **Visual Analytics**: Track your training progress with detailed loss plots

## 🚀 Quick Start

```python
from roll_to_train import RollToTrain, Wizard, ExperienceSystem

# Initialize your model, optimizer, and scheduler
model = YourTransformerModel()
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Create your trainer with RPG mechanics
trainer = RollToTrain(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    intelligence=15,  # Your model's intelligence stat
    character_class=Wizard(level=1),  # Choose your class
    use_xp_system=True
)

# Train your model
trainer.train(train_dataloader, eval_dataloader)
```

## 🎮 How It Works

### Core Mechanics

1. **Dice Rolls**: Each weight update is determined by rolling virtual dice (d20 by default)
   - 🎯 **Critical Success (20)**: Full loss application
   - 💥 **Critical Failure (1)**: Inverse loss application
   - ⚔️ **Success (≥15)**: Scaled positive update
   - 🛡️ **Failure (<15)**: Scaled negative update

2. **Character Classes**
   - 🧙‍♂️ **Wizard**: Manipulates learning rates with arcane abilities
   - 🗡️ **Rogue**: Applies gradient boosts with sneak attacks
   - (More classes coming soon!)

3. **Monster Encounters**
   - 🐉 **Loss Dragon**: Increases loss values
   - 🦠 **Gradient Ooze**: Slows down updates
   - 👻 **Weight Wraith**: Affects model weights

## 📈 Results

Check out our [experiments](experiments/) directory for detailed results and comparisons with traditional training methods.

## 🤝 Contributing

We welcome contributions! Whether you want to:
- Add new character classes
- Create new monsters
- Implement additional RPG mechanics
- Improve documentation
- Fix bugs

Please check out our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## 📚 Documentation

For detailed documentation, visit our [Wiki](https://github.com/yourusername/roll-to-train/wiki).

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by D&D 5e mechanics
- Built on PyTorch and HuggingFace transformers
- Special thanks to all contributors and supporters

---

<div align="center">
  <sub>Built with ❤️ by the Roll-to-Train team</sub>
</div>

# **Roll-to-Train: A Gamified Approach to Machine Learning Training**

*Roll-to-Train* is a novel, experimental, method for training transformer models that integrates mechanics inspired by tabletop role-playing games (RPGs), 
such as dice rolls and saving throws, into the training process. This approach incorporates randomness not just for initialization or 
sampling but directly into the optimization step, creating a dynamic and gamified training paradigm. 

The potential benefits include increased robustness, creative exploration of optimization landscapes, and an engaging tool for educational purposes.

This repo is in active development, and I am welcome to suggestions and contributions to take this even further!

see the CONTRIBUTING.md file.

---

### **Methodology**

The current implementation is very basic, and only the default training option works. It implements dice-roll mechanics 
into the weight update step on either a `per_mini_batch` or `per_accumulation_step` basis. When applied per mini-batch, 
the loss for each batch is scaled directly in accordance with the outcome of the dice rolls. For each accumulation step, the
stored weight updates for each parameter are scaled in the same way.

1. **Saving Throws**: For each weight update, a random integer between 1 and 20 is rolled:
   - **Critical Failure (1)**: The accumulated loss is discarded, and no update occurs.
   - **Critical Success (20)**: Full loss is used for a scaled weight update.
   - **Above Threshold (e.g., ≥15)**: A partial weight update scaled by the roll result is applied.
   - **Below Threshold (e.g., <15)**: A scaled inverse of the weight update is applied to simulate negative learning.
   
2. **Dynamic Update Scaling**: The magnitude of weight updates is scaled based on the roll, introducing randomness into the training trajectory.

3. **Compatibility**: This method is designed to integrate seamlessly with existing frameworks like PyTorch and HuggingFace, requiring only minor modifications to the optimizer and training loop.

---

### **Potential Applications**
1. **Robustness Testing**: Introducing randomness into weight updates can test a model's resilience to suboptimal optimization.
2. **Educational Tool**: The gamified mechanics offer an engaging way to teach machine learning concepts.
3. **Exploration of Loss Landscapes**: By breaking deterministic patterns, this approach could lead to unexpected paths through the optimization space.


### References

@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

