# Block 4 — Spatial Architecture Design

## 1. Locked Baselines
- **E0 (Historical Montage):** 68.19% LOSO Accuracy
- **E1 (Wearable Montage):** 59.28% LOSO Accuracy
- **Degradation ($\Delta$):** -8.91 percentage points

## 2. Scientific Question
"Is the 8.91 pp degradation primarily caused by the wearable electrodes intrinsically containing less usable auditory attention information, or because the baseline EEGNet spatial operation (linear depthwise convolution) is insufficiently expressive to reconstruct the signal from this specific montage?"

## 3. Hypotheses
- **H1 (Information Bottleneck):** The physical wearable locations (forehead and around the ear) fundamentally lack the high-SNR Auditory Evoked Potentials (e.g., N100, P200) present at the vertex/central dipoles. No architectural change will fully recover the lost performance.
- **H2 (Architecture Bottleneck):** The signal is present in the wearable electrodes, but is highly mixed with ocular/muscular artifacts and requires dynamic, non-linear, or cross-channel spatial mixing to extract. The static, linear depthwise convolution of EEGNet is mathematically incapable of learning this complex mapping.

## 4. What E1 Establishes
- Replacing the historical montage with the `[Fp1, Fp2, F7, F8, P7, P8, TP7, TP8]` wearable montage causes a severe, structural performance collapse (-8.91%) when processed by a standard EEGNet encoder.
- The degradation is heavily concentrated in the "easiest" subjects who previously drove high accuracy, indicating that the most easily decodable spatial features (central dipoles) were removed.
- A naive port of the hardware form-factor onto the historical software pipeline is non-viable.

## 5. What E1 Does Not Establish
- **E1 does NOT establish that the wearable form-factor is useless.** It only proves it fails under a rigid, linear spatial filter.
- **E1 does NOT establish that *all* 8-channel wearable montages are bad.** It only tests this specific subset of 8 peripheral channels.
- **E1 does NOT establish that self-attention or Conformer blocks are required.** A simpler non-linear channel mixer might suffice.

## 6. Spatial Ablation Ladder
To rigorously test whether the architecture or the montage is the bottleneck, we will systematically increase the expressiveness of the spatial operation while holding all other variables constant.

**S0: Baseline Control (EEGNet)**
- **Operation:** Linear Depthwise Convolution (`groups=F1`). Each temporal filter learns a fixed, independent linear weighting of the 8 channels.
- **Purpose:** This is the exact E1 control (59.28%).

**S1: Dense Linear Spatial Mixing (Un-grouped)**
- **Operation:** Replace the depthwise spatial convolution with a standard 2D convolution (`groups=1`) across channels.
- **Purpose:** Allows cross-talk between different temporal frequency filters during spatial mixing. Still a static linear map.
- **Parameter Cost:** Negligible increase (from `F1 * D * C` to `F1 * F1 * D * C`).

**S2: Non-Linear Spatial Mixing (Spatial MLP)**
- **Operation:** Pass the channels through a lightweight 2-layer MLP (1x1 Conv -> GELU -> 1x1 Conv) before or after temporal filtering.
- **Purpose:** Tests if non-linear interactions between electrodes are necessary to isolate the auditory signal from peripheral artifacts.
- **Parameter Cost:** Small increase (~few thousand params).

**S3: Dynamic Spatial Mixing (Channel Attention)**
- **Operation:** Squeeze-and-Excitation (SE) Block or lightweight Self-Attention across the channel dimension (treating each channel as a token).
- **Purpose:** Allows the network to dynamically suppress noisy channels (e.g., blinking on Fp1) and attend to clean channels on a per-window basis.
- **Parameter Cost:** Moderate increase, depending on attention heads and embedding dimension.

**S4: Full Spatio-Temporal Interaction (Conformer/Transformer)**
- **Operation:** Flatten spatial/temporal features and use global self-attention (e.g., the existing MSCA Lightweight Conformer block).
- **Purpose:** Maximum expressivity. Tests if complex, long-range spatio-temporal dynamics are required.
- **Parameter Cost:** High.

## 7. Controlled Variables
For experiments S1 through S3, we will rigidly lock:
- Exact E1 wearable montage (`[0, 33, 6, 41, 22, 59, 15, 52]`)
- Exact preprocessing (1-6 Hz bandpass, Z-score)
- Exact audio representation (Gammatone envelopes)
- Exact MatchNet contrastive loss and training loop
- Exact 10 s non-overlapping evaluation windows
- Exact 18-fold LOSO splits

## 8. Metrics
- Overall 18-fold LOSO Accuracy (%)
- Subject-level $\Delta$ (Experiment - S0)
- Total Parameter Count
- (Optional) Relative inference latency

*Note: The 5,400 windows are NOT treated as independent observations for statistical significance. We rely on the subject-level LOSO folds.*

## 9. Interpretation Matrix
- **No Improvement (Across S1-S4):** Strong evidence for **H1**. The wearable electrodes simply do not capture enough physical auditory attention signal. The form-factor or channel selection is the true bottleneck.
- **Small Improvement (+1-3%):** Suggests minor architectural inefficiencies, but the wearable montage remains a severe limitation.
- **Substantial Improvement (+4-6%):** Evidence for **H2**. The signal is present but highly entangled. The static linear filter of EEGNet was a major bottleneck.
- **Recovery Near E0 (-1 to +1% of 68.19%):** Conclusive proof of **H2**. The wearable montage contains equivalent information to the central montage, but requires non-linear/dynamic decoding.
- **Improvement Beyond E0 (>69%):** The wearable montage + expressive architecture is strictly superior to the historical baseline.

## 10. Confounds and Failure Modes
- **Parameter Confound:** If an architecture improves performance, it might just be because it has 10x more parameters, not because it models spatial features better. We must track parameter counts carefully.
- **Overfitting:** More expressive spatial models (S3, S4) might perfectly memorize the training subjects' spatial topographies and fail catastrophically on the held-out LOSO subject. We must monitor train vs. validation loss gaps.
- **Montage Confound:** If all spatial architectures fail, we still cannot prove *all* wearable montages fail. A future experiment would need to test an alternative wearable subset (e.g., swapping Fp1/Fp2 for more temporal/parietal channels) using the best discovered architecture.

## 11. Recommended First Experiment
**Start with S3 (Dynamic Spatial Mixing / Channel Attention).**
Instead of stepping through S1 and S2 sequentially, we can implement a lightweight, parameter-controlled Channel Attention mechanism (e.g., a Squeeze-and-Excitation or lightweight Cross-Covariance attention). 

If a dynamic spatial filter fails to recover performance, it provides strong immediate evidence that the static/dynamic nature of the spatial filter is *not* the primary bottleneck, and shifts the probability massively toward the physical montage lacking signal. 

## 12. STOP Conditions
- Do not proceed to S4 if S3 shows catastrophic overfitting.
- Do not implement self-attention without matching the parameter scale of the baseline EEGNet.
- Do not change the temporal encoder during these spatial experiments.
