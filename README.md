# **Roll-to-Train: A Gamified Approach to Machine Learning Training**

*Roll-to-Train* is a novel, experimental, method for training transformer models that integrates mechanics inspired by tabletop role-playing games (RPGs), 
such as dice rolls and saving throws, into the training process. This approach incorporates randomness not just for initialization or 
sampling but directly into the optimization step, creating a dynamic and gamified training paradigm. 

The potential benefits include increased robustness, creative exploration of optimization landscapes, and an engaging tool for educational purposes.

This repo is in active development, and I am welcome to suggestions and contributions to take this even further!

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
