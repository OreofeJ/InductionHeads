# Research Note: Causal Analysis and Identification of Induction Head Circuits in GPT-2 Small

## 1. Research Question & Hypothesis
* **Question:** Can induction heads be causally identified in GPT-2 Small, and how does disrupting their activations affect in-context pattern copying?
* **Hypothesis:** Induction heads contribute disproportionately to next-token prediction under repeated token sequences, and their causal necessity can be established via targeted multi-head ablation.

## 2. Methodology Summary
* **Model:** GPT-2 Small via `TransformerBridge` (compatibility mode).
* **Dataset:** 1,000 synthetic examples of repeated token sequences `[A, B, C, ..., A, B]` with target `C` (Seed: 77).
* **Identification:** Extracted post-softmax attention patterns (`hook_pattern`) across all 144 heads to calculate induction scores.
* **Causal Validation:** Performed single-head activation patching, single-head zero ablation, multi-head ablation (Top 5 induction heads), and negative control ablation (5 random heads).

## 3. Key Findings & Empirical Results
* **Induction Score Peak:** Identified strong induction activity concentrated in layers 5 through 7 (e.g., Layer 5, Heads 5 & 1; Layer 6, Head 9; Layer 7, Head 10).
* **Single-Head Ablation:** Zero-ablating a single top head (Layer 5, Head 5) yielded **92.00% accuracy**, demonstrating redundancy and distributed representation across the circuit.
* **Multi-Head Circuit Necessity:** Zero-ablating the top 5 candidate induction heads simultaneously caused a substantial performance collapse:
  * **Accuracy:** Dropped to **63.00%**
  * **Mean Logit Difference:** Dropped to **-0.0709** (indicating the model actively favored incorrect competing tokens).
* **Negative Control:** Ablating 5 random non-induction heads yielded **96.00% accuracy** and a robust logit difference of **+3.6859**, confirming that performance drops are structurally specific to the induction circuit rather than general capacity loss.

## 4. Circuit Limitations & Where the Explanation Fails
1. **Redundancy & Distributed Burden:** Induction behavior is not localized to a single "switch." While the top 5 heads account for a sharp drop, remaining heads and earlier previous-token heads share the computational load, meaning absolute circuit necessity cannot be pinned to an isolated subset without broader degradation.
2. **Task Generalization:** Synthetic repeated sequences isolate copying behavior cleanly, but performance under natural text distributions with noise and variable sequence lengths shows that induction heads interact heavily with MLPs and earlier token-retrieval heads, complicating clean modular separation.
