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
* **Use shared library** when possible:
  ```python
  from aiml_notebooks import CharacterTokenizer, create_dataset, create_dataloaders, get_device, set_seed

  %load_ext autoreload
  %autoreload 2
  ```
* **Set random seed** early: `set_seed(42)`
* **Device management**: `device = get_device()` or `device = get_device(prefer_cpu=True)` for Transformers
* **Import organization**: Group stdlib → third-party → local library

---

### 📐 Notebook Template Structure

```markdown
# Title: Clear, Descriptive Name

## 1. Introduction
- What we'll build/learn
- Why it matters
- What intuitions we'll develop

## 2. Setup
- Imports (with %autoreload)
- Random seeds
- Device setup

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

#### **Pattern 1: Introducing a Concept**

```markdown
### Understanding Cross-Entropy Loss

The **cross-entropy loss** measures how well our predicted probability distribution matches the true distribution. Lower values mean better predictions.

Intuitively: it heavily penalizes confident wrong predictions.
```

```python
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

#### **Pattern 2: Building Incrementally**

```markdown
Let's build a simple neural network step by step.

First, a **single linear layer** — the simplest possible network.
```

```python
class SimpleNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet(10, 3)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

```markdown
Now add a **hidden layer** to learn non-linear patterns.
```

```python
class BetterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = BetterNet(10, 20, 3)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

```markdown
The hidden layer adds capacity — but also more parameters to train. This is the classic **capacity vs. complexity** tradeoff.
```

#### **Pattern 3: Visualization-Driven Learning**

```markdown
### How Does Learning Rate Affect Training?

Let's train the same model with different learning rates and visualize the loss curves.
```

```python
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

* **Code without context** — never have a code cell without a preceding markdown cell
* **Large code dumps** — break into smaller, digestible pieces
* **Skipping visualizations** — always show, don't just tell
* **Assuming knowledge** — explain every new term
* **Non-executable cells** — test end-to-end before finalizing
* **Library API errors** — verify methods exist (e.g., `tokenizer.chars` not `tokenizer.get_vocab()`)
* **Forgetting device** — always move tensors/models to device
* **No random seed** — results should be reproducible

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

- [ ] Every code cell has a markdown cell before it
- [ ] Random seed is set early
- [ ] Device is configured properly
- [ ] Visualizations appear after introducing new concepts
- [ ] Each section builds on the previous one
- [ ] No API calls to non-existent methods
- [ ] Notebook runs end-to-end without errors
- [ ] 50-80 cells total
- [ ] Clear narrative flow from start to finish
- [ ] Key insights are explicitly stated

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
