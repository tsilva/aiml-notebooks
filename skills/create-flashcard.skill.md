## 🧠 **Flashcard Creation Prompt (Final Version)**

**Instruction to AI:**
Create **atomic flashcards** and save each as a separate `.md` file in the `flashcards/` folder.

---

### ⚙️ Rules

* **One concept only** → one question, one answer.
* **Minimal and direct** → no filler, no context.
* **Use LaTeX** for all math (`$inline$` or `$$display$$`).
* **Bold key terms** → highlight important terms (e.g., **MSE**, **precision**, **overfitting**, **bias**) in both questions and answers.
* **Answer ≤ 3 sentences** (preferably one line).
* **When the answer is a formula**, include a **glossary** listing each variable briefly (1–2 words each).
* **Exact format below** — no extra text, no headings.
* **Output folder:** `flashcards/`
* **File name:** short, descriptive, related to the question (e.g., `mse_gradient.md`, `dropout_purpose.md`).
* **Default behavior:** create **one flashcard** per run.
* **If multiple flashcards requested:**

  * Each flashcard must follow the same format.
  * Each flashcard saved as a **separate file** in `flashcards/`.
  * **One atomic question–answer pair per file** — no grouping.

---

### 📄 Format

```markdown
Question?
---
$$Formula$$

- $x$: description  
- $y$: description  
- $z$: description
```

---

### ✅ Example (with formula)

```markdown
What is the gradient of the **mean squared error (MSE)** loss with respect to predictions?
---
$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n} (\hat{y} - y)$$

- $L$: loss
- $\hat{y}$: predicted values
- $y$: true target values
- $n$: number of samples
```

---

### 💬 Example (non-formula)

```markdown
What is the purpose of **dropout** in neural networks?
---
Reduces **overfitting** by randomly deactivating neurons during training.
```

---

**Remember:**

> "One concept, one question, one minimal answer — with a glossary if it's a formula."
> "**Bold key terms** in questions and answers for emphasis."
> Each flashcard = one `.md` file in `flashcards/`.
