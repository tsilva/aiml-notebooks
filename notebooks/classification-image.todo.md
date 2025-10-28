🧭 PyTorch Lightning Image Classification Learning Roadmap


- Overfit batch, ask best way to do this
- Run MLP dimension sweep: `uv run python sweep.py sweeps/classification-image-mlp.yaml notebooks/classification-image.ipynb`
- Print model architecture before starting

1️⃣ Lightning Warm-Up

Dataset: MNIST
Model: 2-layer MLP (no convs)
Target: ≥ 97% test accuracy
Learn:
	•	How to define a LightningModule and Trainer.
	•	Overfit on one batch to verify setup.
	•	Logging with self.log().
💡 Side quest: visualize sample batches with matplotlib.

⸻

2️⃣ First CNN

Dataset: MNIST
Model: Simple CNN (2 conv blocks → FC)
Target: ≥ 99%
Learn:
	•	Build your own small ConvNet.
	•	Add LightningDataModule for clean data loading.
	•	Track accuracy via torchmetrics.
💡 Side quest: try different activation functions (ReLU vs LeakyReLU).

⸻

3️⃣ Regularization & Augmentation

Dataset: Fashion-MNIST
Model: CNN + Dropout + BatchNorm
Target: ≥ 93%
Learn:
	•	Data augmentation with torchvision.transforms.
	•	Early stopping & checkpoint callbacks.
	•	Overfitting vs underfitting dynamics.
💡 Side quest: plot training vs validation loss curves.

⸻

4️⃣ Real-World Digits

Dataset: SVHN
Model: 3–4 Conv blocks CNN
Target: ≥ 96%
Learn:
	•	Normalization per channel.
	•	Experiment logging (TensorBoard/W&B).
	•	Handling class imbalance if it appears.
💡 Side quest: visualize misclassified images.

⸻

5️⃣ Transfer Learning Starter

Dataset: Imagenette (subset of ImageNet)
Model: Pretrained ResNet18 (freeze → unfreeze)
Target: ≥ 95%
Learn:
	•	Using pretrained weights (torchvision.models).
	•	Finetuning vs feature extraction.
	•	Mixed precision training (precision="16-mixed").

⸻

6️⃣ Training from Scratch

Dataset: CIFAR-10
Model: ResNet18 (scratch)
Target: ≥ 90%
Learn:
	•	Strong augmentation (RandomCrop, Flip, Normalize).
	•	LR schedulers & learning rate finder.
	•	Model checkpoints, resume training.

⸻

7️⃣ Advanced Training Tricks

Dataset: CIFAR-10
Model: WideResNet-28-10 or ResNet34
Target: ≥ 94%
Learn:
	•	CutMix / MixUp, label smoothing.
	•	Large-batch scaling & cosine warmup.
	•	Exponential moving average (EMA) weights.

⸻

8️⃣ Domain Shift & Generalization

Dataset: STL-10
Model: Finetuned ResNet34
Target: ≥ 80%
Learn:
	•	Transferring to higher-res (96×96 → 224×224).
	•	Evaluating on new distributions.
	•	Tracking per-class metrics.

⸻

9️⃣ Small Data Discipline

Dataset: Oxford-IIIT Pet
Model: Pretrained EfficientNet-B0 or ConvNeXt-Tiny
Target: ≥ 90%
Learn:
	•	K-fold validation or data splits.
	•	Heavy augmentation vs overfitting.
	•	Weighted losses for class imbalance.

⸻

🔟 Scaling to Medium Data

Dataset: EuroSAT (RGB)
Model: Pretrained ResNet50
Target: ≥ 97%
Learn:
	•	Adjusting LR & batch size for GPU memory.
	•	Detailed validation metrics and confusion matrix.

⸻

11️⃣ Bigger Class Space

Dataset: Tiny-ImageNet (200 classes)
Model: ResNet34 / DenseNet121
Target: ≥ 60%
Learn:
	•	Long training schedules.
	•	Efficient augmentations.
	•	Saving & averaging checkpoints.

⸻

12️⃣ Reproducibility

Dataset: CIFAR-10 (repeat)
Model: Your best model so far
Target: Stable (±0.3%) across 3 seeds
Learn:
	•	Seeding and deterministic training.
	•	Exporting configs (LightningCLI, hparams.yaml).
	•	Automating experiments reproducibly.

⸻

13️⃣ Robustness & Error Analysis

Dataset: CIFAR-10-C (corrupted variants)
Model: Best CIFAR model
Target: ≤ 10 pp drop under mild noise
Learn:
	•	Evaluating robustness and calibration.
	•	Reliability diagrams, top-k metrics.

⸻

14️⃣ Deployment Basics

Dataset: Imagenette or CIFAR-10
Model: Quantized / pruned ResNet18
Target: ≤ 1 pp accuracy drop, ≥ 2× smaller
Learn:
	•	TorchScript, ONNX export, CPU inference.
	•	Measuring latency and model size.

⸻

15️⃣ Stretch Goal: Fine-Grained or Multi-Label

Dataset: Caltech-101 (fine-grained) or Pascal VOC 2007 (multi-label)
Model: ConvNeXt-Tiny, BCEWithLogitsLoss
Target: Caltech ≥ 90%, VOC mAP ≥ 85%
Learn:
	•	Adapting losses and metrics.
	•	Threshold tuning, Grad-CAM visualization.

⸻

✅ Always check
	•	Can your model overfit one batch?
	•	Are data augmentations applied correctly?
	•	Are metrics logged to TensorBoard/W&B?
	•	Are seeds fixed for reproducibility?
	•	Are test results reproducible and saved?

⸻

Would you like me to turn this roadmap into a visual progress tracker (markdown table or printable checklist) so you can tick goals off as you go?