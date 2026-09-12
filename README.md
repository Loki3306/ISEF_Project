# 🧠 ISEF Project: MSCA-AAD

## Multi-Scale Attention-Augmented Domain-Adversarial Neural Decoding for Neuro-Steered Hearing Aids

### 🎯 Mission
Develop next-generation, high-speed Auditory Attention Decoding (AAD) for wearable/in-ear EEG interfaces operating on ultra-short decision windows (0.1s – 1.0s), combining multi-scale temporal convolutions, learnable SincNet bandpass filterbanks, channel squeeze-and-excitation attention, lightweight Conformer modules, dual-head decision fusion, and domain-adversarial gradient reversal (GRL) for robust Leave-One-Subject-Out (LOSO) generalization.

---

### 📐 Architecture Blueprint (MSCA-AAD)

```text
                                 EEG Input (8 × T @ 64Hz)
                                            │
                             ┌──────────────┴──────────────┐
                             │                             │
                   TEMPORAL MULTI-SCALE             SPECTRAL SINC-CONV
                     (k = 3, 7, 15, 31)             (8 Learnable Bands)
                             │                             │
                             └──────────────┬──────────────┘
                                            │
                                  True Spatial Topology
                                            │
                                  SE Channel Attention
                                            │
                                    Temporal Pool /4
                                            │
                                Lightweight Conformer ×2
                                            │
                                    Temporal Pooling
                                            │
                                   z_E (64-D Latent)
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
            MatchNet Head           Direct Classifier       GRL Subject Head
          (Cosine Similarity)             (MLP)            (Domain Invariant)
                    │                       │                       │
                    └───────────┬───────────┘                Loss: L_domain
                                │
                        Learned Gate Fusion
                                │
                         P(A), P(B) Logits
                                │
                        Loss: L_CE, L_margin
```

---

### 📂 Repository Structure

```text
ISEF_Project/
├── architecture_plan_1.md      # Foundational architecture specification and engineering audits
├── README.md                   # Project overview and roadmap
├── config/                     # Hyperparameters, sampling rates, and channel topologies
├── models/                     # PyTorch implementations (MSCA-AAD modules, Conformer, SincConv)
├── datasets/                   # EEG & Audio dataset loaders (AASD, KUL, DTU pipelines)
├── training/                   # Multi-stage curriculum training runners & losses
└── evaluation/                 # LOSO benchmark runners (0.1s to 10s), falsification tests
```

---

### 📑 Key Documents
* Detailed Architecture & Design Critique: [`architecture_plan_1.md`](architecture_plan_1.md)
