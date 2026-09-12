# Block 4 — Spatial Architecture Design

## 1. Locked Baselines
- **E0 (Historical Montage):** 68.19% LOSO Accuracy (`[13, 46, 43, 23, 50, 0, 52, 14]`)
- **E1 (Wearable Montage):** 59.28% LOSO Accuracy (`[0, 33, 6, 41, 22, 59, 15, 52]`)
- **Degradation ($\Delta$):** -8.91 percentage points

## 2. Scientific Question
"Is the 8.91 pp degradation primarily caused by the wearable electrodes intrinsically containing less usable auditory attention information, or because the baseline EEGNet spatial operation (static linear depthwise convolution) is insufficiently expressive to reconstruct the signal from this specific montage?"

## 3. Hypotheses
- **H1 (Information Bottleneck):** The physical wearable locations (forehead and around the ear) fundamentally lack high-SNR Auditory Evoked Potentials.
- **H2 (Architecture Bottleneck):** The signal is present in the wearable electrodes, but it is highly entangled with ocular/muscular artifacts and requires non-linear or dynamic cross-electrode spatial mixing to extract. The static, linear depthwise convolution of EEGNet is mathematically incapable of learning this complex mapping.

## 4. What E1 Establishes
- Replacing the historical montage with this specific wearable montage causes a severe, structural performance collapse (-8.91%) when processed by a standard EEGNet encoder.
- A naive port of the hardware form-factor onto the historical software pipeline is non-viable.

## 5. What E1 Does NOT Establish
- **It does NOT establish that all 8-channel wearable montages are inferior.** E1 only tested a single specific subset.
- **It does NOT establish that the wearable electrodes lack the necessary information.**
- **It does NOT establish that EEGNet spatial modeling is the root cause of the degradation.**

---

## 6. Spatial Ablation Ladder

### S0 — E1 CONTROL (BASELINE)
- **Operation:** Current EEGNet spatial operation (Linear Depthwise Convolution).
- **Exact Implementation:** `nn.Conv2d(F1, F1*D, kernel_size=(8,1), groups=F1)`
- **Input:** `[B, F1, 8, T]`
- **Output:** `[B, F1*D, 1, T]`
- **Performance:** 59.28%

### S0+ — CAPACITY CONTROL
- **Operation:** A higher-capacity version of the EXISTING EEGNet-style spatial mechanism (static linear).
- **Purpose:** Determine whether any later improvement is simply caused by giving the network more parameters/capacity rather than changing the *type* of spatial operation.
- **Implementation:** Scale `D` or `F1` (or add multiple linear layers that collapse to linear) to overlap the parameter count of S1/S2.

### S1 — STATIC NONLINEAR CROSS-ELECTRODE MIXING
- **Operation:** The FIRST actual spatial hypothesis experiment. A static, non-linear mapping that acts explicitly on the 8 physical electrodes without unintentionally mixing the `F1` temporal feature dimension.
- **Purpose:** "Can a nonlinear static mapping of the eight wearable electrodes extract information that the baseline linear EEGNet spatial filter cannot?"

### S2 — DYNAMIC ELECTRODE ATTENTION
- **Operation:** Treat the 8 physical electrodes as spatial tokens. Attention operates strictly across ELECTRODES, not time.
- **Purpose:** Test whether input-dependent spatial weighting helps suppress noise and recover attention features.
- **Note:** Must be deliberately small. Only tested after S1.

### S3 — ADVANCED SPATIO-TEMPORAL MODEL
- **Operation:** A contingency experiment only. Complex spatio-temporal modeling (e.g., Conformer). 
- **Purpose:** Only to be considered if S1/S2 provide evidence that highly expressive spatial modeling is useful.

---

## 7. Parameter Control
For every model, we will explicitly calculate and report:
- Spatial parameters
- Total trainable parameters
- Parameter increase relative to S0
- Approximate FLOPs/inference cost

We will not claim that S1/S2 superiority proves the non-linear operation itself is better if the model is substantially larger without comparing against S0+. The goal is to make the capacity confound visible and controlled.

---

## 8. Controlled Variables
For every experiment (S1-S3), we rigidly lock:
- Exact E1 wearable channels
- Exact preprocessing
- Exact audio features
- Exact temporal frontend
- Exact loss
- Exact optimizer
- Exact training windows
- Exact LOSO folds
- Exact validation split
- Exact evaluation windows
- Exact metric
- Random-seed policy

---

## 9. Parameter-Matched Interpretation Matrix

| Case | Result | Interpretation |
| :--- | :--- | :--- |
| **A** | S1 > S0 and S1 ≈ S0+ | Improvement may primarily reflect increased capacity, not the non-linear spatial mixing itself. |
| **B** | S1 > S0+ as well | Stronger evidence that non-linear spatial mixing across electrodes is genuinely useful. |
| **C** | S1 ≈ S0 | No evidence that static non-linear spatial mixing helps. |
| **D** | S2 > S1 / S0+ | Evidence that dynamic (input-dependent) cross-electrode weighting helps. |
| **E** | S1/S2 improve but remain far below E0 | Architecture helps extract signal, but the physical wearable montage remains a substantial information bottleneck. |
| **F** | S1/S2 approach or exceed E0 | Strong evidence that substantial information is present in the wearable montage, and the baseline spatial modeling was the major limitation. |

*Note: If increasingly expressive cross-electrode models fail to improve performance under controlled conditions, the evidence shifts toward the wearable montage/electrode information being a dominant bottleneck, but this does NOT prove that the information is absent.*

---

## 10. Montage Confound
**Crucial limitation:** E1 tested only `[Fp1, Fp2, F7, F8, P7, P8, TP7, TP8]`. A failure to recover performance across S1-S3 does not establish that all 8-channel wearable montages are inferior. A later experiment must test alternative 8-channel montages. We do not perform that experiment now, to keep Block 4 controlled.

---

## 11. Distinction from ATCNet
The repository documents ATCNet at ~64.89%. ATCNet uses temporal attention and TCNs *after* the spatial dimension has already been collapsed using the exact same static, linear depthwise spatial filter as EEGNet. ATCNet is a temporal mixing model. Block 4 tests **cross-electrode spatial mixing**. We must not simply reproduce ATCNet.

---

## 12. FIRST EXPERIMENT DETAILS: S1 (Static Nonlinear Cross-Electrode Mixing)
The recommended first implementation is **S1**, not attention.

1. **Exact Tensor Shapes:**
   - Input to spatial block: `[B, F1, 8, T]`
   - Reshape to isolate the electrode dimension: `[B, F1, T, 8]`
   - MLP operates on the last dimension (`8`), outputting `D`.
   - Output reshape: `[B, F1, D, T]` -> `[B, F1*D, 1, T]`
2. **Exact Operation:**
   - Flatten B, F1, T into the batch dimension.
   - `Linear(in_features=8, out_features=Hidden_Dim)`
   - `GELU()`
   - `Linear(in_features=Hidden_Dim, out_features=D)`
3. **Exact Location in EEGNet:** Replaces `self.block1`'s spatial `Conv2d` depthwise layer.
4. **How the electrode dimension is preserved:** By applying the dense layers strictly over the `8` electrode dimension, treating `F1` and `T` as independent batch elements.
5. **How the spatial dimension is ultimately collapsed:** The final linear layer projects from `Hidden_Dim` to `D`, meaning the 8 electrodes are non-linearly collapsed into `D` spatial representations for each of the `F1` temporal filters.
6. **Parameter Count:** `8 * Hidden_Dim + Hidden_Dim * D` (plus biases).
7. **Expected Parameter Range:** Small (~100 to 1,000 parameters depending on `Hidden_Dim`).
8. **S0+ Capacity-Control Design:** A linear static depthwise convolution that uses a much larger `D` value to match the parameter count of the S1 MLP, and then projects back down.
9. **Frozen Variables:** Everything outside the spatial block.
10. **Exact Evaluation Protocol:** 18-fold LOSO, 10s non-overlapping windows.
11. **Interpretation:** Handled by the matrix in Section 9.
12. **STOP Condition:** Do not proceed to S2 without reviewing S1 against S0+.

---

## 13. STOP CONDITIONS (Global)
- Do not allow architecture creep.
- STOP after S1 design. Do not proceed automatically to S2.
- Do not change: temporal encoder, frequency preprocessing, audio representation, loss, windowing, channels, or LOSO protocol.
