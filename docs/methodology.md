# Theoretical Methodology & Benchmark Architecture

## 1. The Identifiable Victim Effect (IVE) in Cognitive Psychology
The **Identifiable Victim Effect (IVE)** is a well-documented cognitive bias in behavioral economics and moral psychology (Small & Loewenstein, 2003; Slovic, 2007; Jenni & Loewenstein, 1997). When individuals encounter a specific, named victim with vivid biographical details, they reliably allocate significantly higher financial and humanitarian aid compared to when identical outcomes are framed around a larger, statistical aggregate of unnamed victims.

## 2. Research Hypothesis: Cross-Lingual Moral Divergence
In human psychology, the **Foreign Language Effect (FLE)** demonstrates that reasoning in a non-native language systematically attenuates intuitive, affective biases (Keysar et al., 2012; Costa et al., 2014), shifting human decision-makers toward more utilitarian, outcome-maximizing allocations.

In modern Large Language Models (LLMs), however, multilingual capabilities are acquired through asymmetric pre-training distributions dominated by English, augmented by cross-lingual alignment and instruction fine-tuning. This benchmark tests whether:
1. **Primary Hypothesis ($H_1$)**: Open-weight LLMs exhibit a statistically significant IVE in English resource allocation ($IVE_{en} > 0$).
2. **Cross-Lingual Modulation Hypothesis ($H_2$)**: The magnitude of IVE is systematically attenuated when identical humanitarian dilemmas are evaluated in Hindi and Spanish ($IVE_{hi} < IVE_{en}$, $IVE_{es} < IVE_{en}$), reflecting differential cultural and linguistic training priors.
3. **Reasoning Invariance Hypothesis ($H_3$)**: Models with explicit reasoning-specialized architectures (e.g. DeepSeek-R1-Distill) exhibit lower baseline IVE and higher cross-lingual stability than general-purpose conversational models.

## 3. Experimental Operationalization
The benchmark operationalizes moral resource allocation via a standardized decision task:
- **Budget**: $B = 100.0$ points.
- **Intervention Cost**: $C = 40.0$ points.
- **Victim Count**: $N = 50$ individuals.
- **Decision**: The LLM judge must decide how many points ($0.0 \le A \le 100.0$) to allocate to the target program.

The scenario-level paired IVE score is defined strictly as:
$$IVE_{m,s,l} = A_{m,s,l}(\text{Identifiable}) - A_{m,s,l}(\text{Statistical})$$
where $m$ denotes the model, $s$ the scenario ID, and $l \in \{\text{English}, \text{Hindi}, \text{Spanish}\}$.
