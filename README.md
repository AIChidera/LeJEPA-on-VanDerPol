# LeJEPA on Van der Pol Oscillator

A proof-of-concept implementation of **LeJEPA** (Latent-space Joint-Embedding Predictive Architecture) applied to a synthetic physics dataset. My goal was to simply train a single encoder to predict the future states of a nonlinear oscillator, with no frozen target network, no stop-gradient, and no exponential moving average. Then check whether the latent space actually learned the geometry of the system.

It did.

---

## What is LeJEPA?

Most self-supervised learning architectures that predict in latent space (like I-JEPA or data2vec) rely on two encoders: one "context" encoder that gets updated by gradients, and a "target" encoder that is a slowly updated copy of the first. The target encoder is frozen on purpose. Without it, the model collapses, because the easiest way to make two representations match is to make both of them the same constant.

LeJEPA replaces that two-encoder setup with a single encoder and a regularization term called **SIGReg** (Sketched Isotropic Gaussian Regularization). SIGReg directly penalizes the model whenever the latent representations start collapsing toward each other. It does the same job as the EMA target network, but with one network and one hyperparameter (`lambda`).

This experiment tests that claim on a controlled problem where the ground truth geometry is known.

---

## The Van der Pol Oscillator

The Van der Pol oscillator is a nonlinear differential equation that describes systems with self-sustaining oscillations, things like certain electronic circuits and biological rhythms. Its defining property is a **stable limit cycle**: no matter where you start in phase space, the trajectory spirals in or out until it lands on the same closed loop.

```
dx/dt = y
dy/dt = (1 - x²)y - x
```

This makes it a perfect test case. If the model genuinely learns the dynamics, its latent space should also form a closed ring, because the underlying physics is a closed ring. If the model collapses or learns nothing useful, the plot will show a blob or a line.

---

## Architecture

The model has three parts:

**Encoder** — A two-layer 1D CNN with GELU activations and LayerNorm, projecting raw 2D sequences into a 64-dimensional embedding space.

**Predictor** — A two-layer MLP that takes the flattened context embeddings (first 70 timesteps) and predicts the target embeddings (last 30 timesteps).

**SIGReg** — The Epps-Pulley test wrapped in a slicing framework with 256 random projections, operating in O(N) time. This is the collapse prevention mechanism.

The forward pass uses **block masking**: the encoder sees the entire sequence in one shot, then the embeddings are logically split into context and target. No masking tokens. No separate forward passes. The predictor never sees the target directly.

---

## Training

| Setting | Value |
|---|---|
| Dataset | 5,000 Van der Pol sequences |
| Sequence length | 100 timesteps |
| Context / Target split | 70 / 30 |
| Latent dimension | 64 |
| SIGReg weight (λ) | 0.05 |
| Optimizer | AdamW, lr = 1e-3 |
| Epochs | 20 |
| Hardware | NVIDIA T4 (Google Colab) |

Training logs:

```
Starting LeJEPA Predictive Training...
Epoch 01 | Total: 236.7318 | MSE: 0.5922 | SIGReg: 4722.7927
Epoch 02 | Total: 153.1084 | MSE: 0.1254 | SIGReg: 3059.6603
Epoch 03 | Total: 125.0167 | MSE: 0.0636 | SIGReg: 2499.0614
Epoch 04 | Total: 105.2131 | MSE: 0.0559 | SIGReg: 2103.1441
Epoch 05 | Total: 90.9768 | MSE: 0.0490 | SIGReg: 1818.5562
Epoch 06 | Total: 83.0134 | MSE: 0.0401 | SIGReg: 1659.4656
Epoch 07 | Total: 80.5140 | MSE: 0.0397 | SIGReg: 1609.4842
Epoch 08 | Total: 77.7068 | MSE: 0.0465 | SIGReg: 1553.2042
Epoch 09 | Total: 77.9802 | MSE: 0.0449 | SIGReg: 1558.7071
Epoch 10 | Total: 78.3379 | MSE: 0.0446 | SIGReg: 1565.8658
Epoch 11 | Total: 78.3023 | MSE: 0.0419 | SIGReg: 1565.2086
Epoch 12 | Total: 77.8607 | MSE: 0.0553 | SIGReg: 1556.1071
Epoch 13 | Total: 76.5172 | MSE: 0.0351 | SIGReg: 1529.6430
Epoch 14 | Total: 78.1522 | MSE: 0.0336 | SIGReg: 1562.3719
Epoch 15 | Total: 76.7702 | MSE: 0.0378 | SIGReg: 1534.6477
Epoch 16 | Total: 77.5010 | MSE: 0.0355 | SIGReg: 1549.3098
Epoch 17 | Total: 77.9179 | MSE: 0.0330 | SIGReg: 1557.6997
Epoch 18 | Total: 76.6406 | MSE: 0.0326 | SIGReg: 1532.1595
Epoch 19 | Total: 76.1041 | MSE: 0.0333 | SIGReg: 1521.4150
Epoch 20 | Total: 78.4184 | MSE: 0.0285 | SIGReg: 1567.7967
```

The MSE drops from 0.59 to 0.028 over 20 epochs. The SIGReg dropped from 4722 to about 1550 and stayed there. The fact that it plateaued and held steady in the 1500s means the encoder established a diverse, stable Gaussian distribution and violently defended it against the predictor's attempts to collapse the space.

---

## Results

After training, I extracted the encoder's output at the final context timestep (t=69) for every sequence in the dataset, then reduced from 64 dimensions to 2 using PCA.

![LeJEPA Latent Space Geometry](https://github.com/AIChidera/LeJEPA-on-VanDerPol/blob/cd19714c467c7b06e5fa46d0b113d3fd5ce627ce/Umap%20of%20LeJEPA%20on%20Van%20der%20Pol%20limit%20cycle.png)

**What you are looking at:** each point is one sequence's latent representation. The encoder was never told what a limit cycle is. It was only told to predict the next 30 timesteps from the previous 70. The ring emerged on its own.

Three things this result confirms:

1. **No collapse.** A collapsed model produces a dot or a line. The points cover the full extent of the plot in a structured loop. The SIGReg constraint kept the 64-dimensional space diverse throughout training.

2. **The physics is in there.** The ring in latent space mirrors the limit cycle in phase space. The model built an internal map of the oscillator's dynamics without any supervision signal about what the dynamics are.

3. **Single-encoder JEPA works.** The EMA target network is not the only way to prevent collapse in latent-prediction architectures. A regularization penalty on the geometry of the learned representations is enough, at least at this scale.

---

## Why I Built This

I have been working on JEPA-based architectures for anomaly detection in industrial and physical systems, including a paper on I-JEPA representations for unsupervised anomaly detection (MVTec AD / VisA datasets). LeJEPA caught my attention because it removes the EMA complexity while keeping the core predictive learning objective. Before using it in a larger system, I wanted to understand exactly what it learns and how the training signal behaves.

The Van der Pol oscillator is a good testbed because the answer is known. There is no ambiguity about what a successful result should look like. If the ring shows up, the architecture is working as described.

---

## How to Run

```
# Open the notebook
jupyter notebook LeJEPA_on_Van_der_Pol.ipynb
```

---

## Dependencies

- `lejepa` (galilai-group, GitHub)
- `torch >= 2.0`
- `scikit-learn`
- `matplotlib`
- `numpy`

---

## References

- LeJEPA: [galilai-group/lejepa](https://github.com/galilai-group/lejepa)
- I-JEPA: Assran et al., 2023 — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*
- Van der Pol oscillator: B. van der Pol, 1926 — *On relaxation-oscillations*, Philosophical Magazine

---

## Author

**Chidera Achinike**  
MTech in Artificial Intelligence, Vivekananda Global University, Jaipur
