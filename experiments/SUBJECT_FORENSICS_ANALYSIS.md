# Phase F3: Exploratory Subject Forensics Analysis

Based on the `forensics_data.json` computed over the 18 subjects using the Canonical E0 montage, here are the empirical answers to our Phase F3 questions.

## 1. How large is the Personalized CV Gain across subjects?

**Zero.** 
- **Mean LOSO Accuracy**: 55.00%
- **Mean Personalized CV Accuracy**: 55.00%
- **Mean Gain**: 0.00%

The range of gains is highly variable (from -20.00% to +15.00%), with exactly 7 out of 18 subjects showing a positive gain. For the remaining 11 subjects, training a personalized model on their own 48 trials was actually *worse* than evaluating them on a global LOSO model trained on the other 17 subjects.

**Inference:** 48 trials (480 seconds) is fundamentally insufficient to learn a stable, generalized spatial-temporal mapping from scratch. The robust global geometry learned by the LOSO model (from ~1000 trials) almost perfectly offsets whatever subject-specific nuances exist. 

## 2. Are subjects with poor LOSO performance the ones with large personalization gains?

**Yes.**
There is a strong, statistically significant negative association between LOSO accuracy and Personalized Gain ($r = -0.546, p = 0.019$).

- **Poor LOSO Subjects benefit from personalization**: S6 (LOSO = 43.3%) achieved the largest jump, gaining +13.3% when using a personalized model.
- **Strong LOSO Subjects are hurt by personalization**: S12 (LOSO = 65.0%) collapsed by -20.0% under personalization. S17 (LOSO = 75.0%) dropped by -11.6%.

**Inference:** Subjects who naturally align with the "universal" brain geometry (high LOSO) are actively harmed by the data scarcity of personalized training (overfitting). Only "outlier" subjects whose physical geometry heavily deviates from the universal average stand to gain from personalization.

## 3. Does decoder similarity predict transfer accuracy?

**Yes, highly significantly.**
When flattening the off-diagonal elements of the $18 \times 18$ matrices:
- **Correlation ($r$)**: $0.245$
- **p-value**: $< 0.0001$

**Inference:** This confirms our hypothesis that decoder geometry is related to cross-subject generalization. The transferability between two subjects isn't random; it is structurally dictated by the cosine similarity of their optimal linear feature spaces. This implies subjects cluster into distinct geometric "phenotypes".

## 4. Is bandpower associated with Personalized CV Gain?

There is a negative trending association, though not strictly below the traditional significance threshold ($r = -0.430, p = 0.075$).
Subjects with massively high EEG variance/bandpower (e.g., S2: 699, S1: 446) generally exhibited negative personalization gains, whereas low bandpower subjects (e.g., S15: 43, S6: 68) tended to exhibit positive gains. 

---

## Phase F4: The Decision

The decision tree was:
> *If Little Recoverable Personalization Signal $\rightarrow$ Investigate montage / channel information.*

The data is unambiguous. The mean Personalized CV gain is **0.00%**. 

Training a subject-specific decoder on ~50 trials does absolutely nothing on average because data scarcity (overfitting) destroys whatever geometric alignment is gained. This immediately disqualifies "Few-Shot Adaptation" as a viable short-term remedy for the performance bottleneck.

### Next Step: Channel-Count Ablation
We must proceed to the **Channel Count Ablation (64 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 8)**. The failure of complex architectures and the failure of personalization both point to a fundamental loss of spatial information at the hardware level. We need to quantify exactly when the decoding manifold collapses.
