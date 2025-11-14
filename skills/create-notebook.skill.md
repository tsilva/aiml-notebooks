## 📓 **AI/ML Notebook Creation Guide**

**Instruction to AI:**
Create **highly engaging, interactive notebooks** (50-80 cells) that build deep intuitions effortlessly through step-by-step progression.

---

### 🎯 Core Philosophy

**Build knowledge like climbing stairs — each step small, solid, and leading naturally to the next.**

Notebooks should feel like a conversation with an expert teacher who never assumes prior knowledge and always shows before telling.

---

### ⚙️ Rules

#### **Structure & Flow**
* **Always markdown before code** — even if just one explanatory line
* **Theory before practice** — explain "why" before "how"
* **Building blocks approach** — progress incrementally, each concept building on the last
* **50-80 cells total** — enough depth without overwhelming
* **Self-contained** — runnable end-to-end, no external notebook dependencies

#### **Content Design**
* **One concept per section** — don't bundle multiple ideas
* **Progressive visualizations** — show concepts visually as they develop
* **Formulas with context** — always explain variables and intuition
* **Concrete before abstract** — examples before generalizations
* **Interactive exploration** — encourage tweaking parameters

#### **Cell Patterns**
* **Markdown cells:**
  - Introduce what's coming and why it matters
  - Explain the concept or intuition
  - Connect to previous sections
  - Use LaTeX for math (`$inline$` or `$$display$$`)
  - Bold key terms (e.g., **embedding**, **cross-entropy**, **attention**)

* **Code cells:**
  - One logical unit per cell (e.g., model definition, training loop, visualization)
  - Clear variable names
  - Comments for non-obvious logic only
  - Print/plot results immediately after computation

#### **Technical Requirements**
* **CONFIG dictionary first**: Define all hyperparameters at the top for easy experimentation
  ```python
  CONFIG = {
      # Reproducibility
      'seed': 42,  # Random seed for reproducibility

      # Data
      'batch_size': 128,  # Number of samples per training batch
      'num_workers': 0,  # Number of worker processes for data loading

      # Training
      'learning_rate': 0.001,  # Optimizer learning rate
      'max_epochs': 20,  # Number of training epochs

      # Model
      'num_classes': 10,  # Number of output classes
  }
  ```
* **PyTorch Lightning by default**: Use Lightning unless explicitly building training loop from scratch
  - Models inherit from `L.LightningModule`
  - Implement `training_step`, `validation_step`, `configure_optimizers`
  - Use `L.Trainer` instead of manual training loops
  - **Exception**: Use from-scratch loops when the goal is to teach the fundamentals of gradient descent, backpropagation, or training mechanics
* **Distributed imports**: Place imports in the same cell as their first usage, not all at the beginning
  - Setup utilities: `from aiml_notebooks import get_device, set_seed` (same cell as CONFIG/set_seed)
  - Data: `from torchvision import datasets, transforms` (same cell as data transforms)
  - Model: `import torch, torch.nn as nn, lightning as L` (same cell as model definition)
  - Visualization: `import matplotlib.pyplot as plt, numpy as np` (same cell as first plot)
  - **Exception**: If an import is 1-2 lines and the cell would be too long, place it in the preceding markdown cell's code block or a minimal import-only cell
* **Use shared library** when possible:
  ```python
  from aiml_notebooks import CharacterTokenizer, create_dataset, create_dataloaders, get_device, set_seed

  %load_ext autoreload
  %autoreload 2
  ```
* **Set random seed** from CONFIG: `set_seed(CONFIG['seed'])`
* **Device management**: `device = get_device()` or `device = get_device(prefer_cpu=True)` for Transformers

---

### 📐 Notebook Template Structure

```markdown
# Title: Clear, Descriptive Name

## 1. Introduction
- What we'll build/learn
- Why it matters
- What intuitions we'll develop

## 2. Setup

### Configuration
[Markdown: Explain this is where all hyperparameters live]
```python
CONFIG = {
    # Reproducibility
    'seed': 42,  # Random seed for reproducibility

    # Data
    'batch_size': 128,  # Number of samples per training batch

    # Training
    'learning_rate': 0.001,  # Optimizer learning rate
    'max_epochs': 20,  # Number of training epochs

    # Model
    'num_classes': 10,  # Number of output classes
}
```

### Random Seed & Device Setup
```python
from aiml_notebooks import get_device, set_seed

set_seed(CONFIG['seed'])
device = get_device()
print(f"Using device: {device}")
```

## 3. Building Blocks (3-5 sections)
Each section follows this pattern:

### 3.X Concept Name

#### Intuition
[Markdown: Explain the "why" and core idea]

#### Implementation
[Code: Minimal working example]

#### Visualization
[Code: Plot/print to show it working]

#### Key Insight
[Markdown: What did we just learn?]

## 4. Putting It Together
- Combine concepts from sections
- Show the full picture
- Compare variations

## 5. Experiments & Exploration
- Ablation studies
- Parameter sensitivity
- Edge cases
- "What if?" questions

## 6. Key Takeaways
- Bullet points of core intuitions
- Visual summary if applicable
- Connections to broader concepts
```

---

### ✅ Example Cell Sequences

#### **Pattern 1: Introducing a Concept (with same-cell imports)**

```markdown
### Understanding Cross-Entropy Loss

The **cross-entropy loss** measures how well our predicted probability distribution matches the true distribution. Lower values mean better predictions.

Intuitively: it heavily penalizes confident wrong predictions.
```

```python
# Import where first used
import torch
import torch.nn.functional as F

# True label: class 1 (one-hot: [0, 1, 0])
true_label = torch.tensor([1])

# Prediction 1: confident and correct
pred_correct = torch.tensor([[0.1, 0.8, 0.1]])
loss_correct = F.cross_entropy(pred_correct, true_label)

# Prediction 2: confident and wrong
pred_wrong = torch.tensor([[0.8, 0.1, 0.1]])
loss_wrong = F.cross_entropy(pred_wrong, true_label)

print(f"Loss (confident & correct): {loss_correct:.3f}")
print(f"Loss (confident & wrong): {loss_wrong:.3f}")
```

```markdown
Notice how the confident wrong prediction has ~6x higher loss — cross-entropy punishes overconfidence in the wrong answer.
```

#### **Pattern 2: PyTorch Lightning Model**

```markdown
### Implement the Model

We'll use **PyTorch Lightning** to keep our code clean and organized.
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class MyModel(L.LightningModule):
    def __init__(self, input_size=10, hidden_size=20, num_classes=CONFIG['num_classes'],
                 learning_rate=CONFIG['learning_rate']):
        super().__init__()
        self.save_hyperparameters()

        # Model architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

```markdown
### Train the Model

Lightning handles all the boilerplate — we just create a Trainer and call fit!
```

```python
model = MyModel()

trainer = L.Trainer(
    max_epochs=CONFIG['max_epochs'],
    accelerator='auto',
    devices=1,
    logger=False,
    enable_progress_bar=True
)

trainer.fit(model, train_loader, val_loader)
```

```markdown
**Why Lightning?**
- Eliminates boilerplate training loops
- Handles device management automatically
- Built-in logging and callbacks
- Easier to maintain and extend
```

#### **Pattern 3: Visualization-Driven Learning (with same-cell imports)**

```markdown
### How Does Learning Rate Affect Training?

Let's train the same model with different learning rates and visualize the loss curves.
```

```python
# Import visualization libraries where first used
import matplotlib.pyplot as plt

learning_rates = [0.001, 0.01, 0.1]
histories = {}

for lr in learning_rates:
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = train(model, optimizer, epochs=50)  # assume train() is defined
    histories[lr] = losses

# Plot
plt.figure(figsize=(10, 5))
for lr, losses in histories.items():
    plt.plot(losses, label=f'LR={lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Rate Comparison')
plt.grid(True, alpha=0.3)
plt.show()
```

```markdown
**Key observations:**
- Too small (0.001): slow convergence
- Too large (0.1): unstable, may diverge
- Just right (0.01): smooth and fast

This is why learning rate is the most important hyperparameter to tune.
```

---

### 🚫 Common Mistakes to Avoid

**Structure:**
* **Code without context** — never have a code cell without a preceding markdown cell
* **Large code dumps** — break into smaller, digestible pieces
* **All imports at top** — place imports in the same cell as their first usage
* **Separate import cells** — imports should be in the same cell as the code that uses them (with rare exceptions for very long cells)
* **Hardcoded hyperparameters** — everything should be in CONFIG
* **CONFIG defined too late** — must come before anything that uses it

**Framework:**
* **Manual training loops** — use Lightning unless teaching fundamentals
* **Missing Lightning methods** — need `training_step`, `validation_step`, `configure_optimizers`
* **Device management in Lightning** — Lightning handles this, don't manually move to device

**Content:**
* **Skipping visualizations** — always show, don't just tell
* **Assuming knowledge** — explain every new term
* **Non-executable cells** — test end-to-end before finalizing
* **Library API errors** — verify methods exist (e.g., `tokenizer.chars` not `tokenizer.get_vocab()`)
* **No random seed** — results should be reproducible
* **Uncommented CONFIG keys** — every config value needs an inline comment

---

### 🎓 Teaching Principles

1. **Concreteness Fading**: Start with specific examples → generalize to concepts
2. **Worked Examples**: Show complete solutions before asking for variations
3. **Interleaving**: Mix concepts after introduction to strengthen connections
4. **Immediate Feedback**: Print/plot results right after computation
5. **Progressive Complexity**: Simple → nuanced → edge cases

---

### 📝 Notebook Checklist

Before finalizing, verify:

**Structure:**
- [ ] CONFIG dictionary defined at top with all hyperparameters (with inline comments)
- [ ] Every code cell has a markdown cell before it
- [ ] Imports appear in the same cell as their first usage (not in separate cells or all at beginning)
- [ ] Random seed is set from CONFIG: `set_seed(CONFIG['seed'])`
- [ ] Device is configured properly

**Framework:**
- [ ] PyTorch Lightning used for training (unless explicitly from-scratch)
- [ ] Model inherits from `L.LightningModule`
- [ ] `training_step`, `validation_step`, `configure_optimizers` implemented
- [ ] Uses `L.Trainer` instead of manual loops

**Content:**
- [ ] Visualizations appear after introducing new concepts
- [ ] Each section builds on the previous one
- [ ] No API calls to non-existent methods
- [ ] 50-80 cells total
- [ ] Clear narrative flow from start to finish
- [ ] Key insights are explicitly stated

**Testing:**
- [ ] Notebook runs end-to-end without errors
- [ ] All hyperparameters can be changed via CONFIG
- [ ] Results are reproducible (seed works)

---

**Test the notebook:**
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb
```

---

**Remember:**

> "The best notebooks make complex ideas feel obvious — one small step at a time."
> "Always markdown before code. Always visualize after implementation. Always explain why before how."
> "Build intuitions, not just implementations."
