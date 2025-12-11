# Conversations Are Long, Emotions Are Subtle
Evolution of Context and Emotion Modeling in Conversational Sentiment Analysis (ECEM-CSA)

DS-GA 1011 · December 10, 2025 · Ellie Wang, Georgios Ioannou, Qiya Huang, Ruokai Gu, Ziyu Qi

![Hybrid model flowchart](images/hybrid-flowchart.svg)

## Why Conversational Emotion Is Hard
Single-utterance emotion classification is already challenging, but conversational emotion recognition raises the difficulty in several important ways. Emotions in dialogue rarely appear in clean, self-contained sentences. A single line like *"It's fine, whatever"* might signal annoyance, resignation, or genuine agreement—its meaning depends almost entirely on the surrounding context and the speaker's intent.

More importantly, emotions in conversation unfold over time. They accumulate across turns, shift gradually, and often blur the boundaries between categories such as frustration, annoyance, and sadness. Yet many existing systems still analyze each utterance in isolation, overlooking the long-range dependencies and speaker-specific patterns that shape how emotions evolve within a dialogue.

This blog explores whether modern neural architectures can better follow these emotional trajectories. Rather than proposing a new architecture from scratch, we revisit three influential ideas in conversational emotion recognition—speaker-specific memory networks (CMN), long-context transformers (Transformer-XL), and emotion-anchored contrastive learning (EACL). Each addresses a different limitation in how models interpret emotional dynamics.

We reproduce all three models on the IEMOCAP benchmark and study where they excel and where they fall short, centered around a guiding question: **Can long-range context modeling and structured emotional representations be combined into a single, more interpretable system?**

## What's New
- **Reproduction**  
  We reproduce CMN, Transformer-XL, and EACL on IEMOCAP and report our measured F1s versus the original papers.
- **Hybrid Model**  
  We propose a hybrid that combines TXL long-context recurrence with EACL emotion anchors; we report our estimated F1 and where it helps most.

## Dataset
All experiments use the **IEMOCAP (Interactive Emotional Dyadic Motion Capture)** database—an **acted**, **multimodal**, **multi-speaker** corpus collected at USC SAIL. It contains **~12 hours** of audiovisual data from **dyadic sessions** where actors perform **improvised** and **scripted** scenarios designed to elicit clear emotional expressions. Each utterance is annotated with both **categorical labels** (e.g. anger, happiness, sadness, neutrality) and **dimensional labels** (valence, activation, dominance), making it a cornerstone resource for studying multimodal, expressive communication.

**Dataset:** IEMOCAP is available at https://sail.usc.edu/iemocap/

![Example conversation from IEMOCAP](images/iemocap_example.png)

**Figure 1.** An example conversation from the IEMOCAP dataset. The dialogue shows a dyadic interaction between two speakers (Woman and Man) with alternating turns. Each utterance is annotated with an emotion label (shown in brackets), demonstrating how emotions are associated with individual conversational turns.

### Scope of the Database
- Recognition and analysis of emotional expression
- Analysis of human dyadic interactions
- Design of emotion-sensitive human-computer interfaces and virtual agents

### General Information
- **Keywords:** Emotional, Multimodal, Acted, Dyadic
- **Language:** English
- **Actors:** 10 total (5 male, 5 female)
- **Emotion elicitation:** Improvisations and scripts

### Available Modalities
- Motion capture: facial data plus head movement/angle
- Speech audio
- Video
- Dialog transcriptions
- Alignment: word-, syllable-, and phoneme-level

### Annotations
- **Segmentation:** sessions manually segmented into utterances
- **Annotators:** each utterance labeled by at least 3 humans
- **Categorical attributes:** anger, happiness, excitement, sadness, frustration, fear, surprise, other, neutral
- **Dimensional attributes:** valence, activation, dominance

### Release Notes
The current release covers **all 10 actors** (~12 hours of data) with detailed audiovisual and text information for each improvised and scripted recording. A previous limited release (2 actors) remains available on request. Access requires a release form; see the IEMOCAP site for details.

## Research Questions
This blog post investigates three key research questions that guide our exploration of conversational emotion recognition:

- **How do CMN, Transformer-XL, EACL, and KET differ in performance on conversational emotion recognition?**  
  We systematically compare these four architectures to understand their relative strengths and weaknesses.
- **How do long-context modeling (Transformer-XL) and contrastive anchor learning (EACL) each contribute to accuracy and representation quality?**  
  We analyze the individual contributions of long-range context modeling and structured emotion representations.
- **Does a hybrid model that combines long-context modeling with emotion anchors improve both accuracy and interpretability?**  
  We explore whether integrating the complementary strengths of Transformer-XL and EACL can achieve superior performance while maintaining interpretability through emotion-anchored representations.

These 3 questions structure our investigation, from individual model analysis to comparative evaluation and finally to hybrid architecture design.

## Conversational Memory Network (CMN)
Early approaches treated each utterance as an isolated unit. The **Conversational Memory Network (CMN)** was among the first architectures to explicitly model conversation history, recognizing that emotional dynamics in dyadic conversations are driven by **emotional inertia** and **inter-speaker emotional influence**.

CMN uses a **memory-based architecture** that maintains separate, **speaker-specific histories**. Unlike **context-free systems** (e.g. **SVM-ensemble** methods) or **LSTM-based approaches** like **bc-LSTM**, CMN uses **memory networks** to efficiently capture and summarize **task-specific details** from **conversation history** using **attention mechanisms**.

![CMN architecture](images/cmn_architecture.png)

**Figure 2.** Architecture of the Conversational Memory Network (CMN).

### Multimodal Feature Extraction
CMN adopts a **multimodal approach**, extracting features from **audio**, **visual**, and **textual** sources.

### Speaker-Specific Memory Cells
For a given utterance \(u_i\), CMN maintains **separate histories for each speaker**, encoded with **GRUs**.

![CMN attention visualization](images/cmn_memory.png)

**Figure 3.** Attention mechanism in CMN across 3 hops, capturing self-emotional dynamics and inter-speaker influences.

### Attention-Based Memory Hops
CMN employs **attention** to identify which **historical utterances** are most relevant for classifying \(u_i\). Memories are merged with the current representation via weighted addition and iterated across multiple hops.

### Key Contributions and Limitations
CMN explicitly models **conversational emotional dynamics** and improves accuracy over prior baselines. Its limitation: a **fixed context window** (typically K=40), which struggles with very long conversations where emotional context extends beyond the window.

### Our Reproduction and Analysis
On IEMOCAP, we observed performance aligning with the original ~77.6% weighted accuracy. CMN benefits emotions like **happiness** and **anger**, but struggles with long-range context, motivating extended-context models.

## Transformer-XL
### The Core Problem: Context Fragmentation
Standard transformers operate on fixed-length segments, causing **context fragmentation** when conversations exceed that length.

### Transformer-XL's Solution: Segment-Level Recurrence
Transformer-XL introduces **segment-level recurrence**: cached hidden states from prior segments are reused, growing effective context length while keeping backprop local.

![Architecture comparison: CMN baseline vs CMN + Transformer-XL](images/txl.png)

**Figure 4.** Architectural comparison between CMN baseline and CMN enhanced with Transformer-XL.

### Relative Positional Encodings
Transformer-XL replaces absolute positional encodings with **relative positional encodings**, preventing temporal confusion when reusing cached states.

### Integration with CMN
- Track individual speaker states (CMN)
- Extend each speaker's memory across longer history (Transformer-XL recurrence)
- Apply relative positional encodings for temporal coherence

### Impact
Transformer-XL improves **long conversations** where emotions shift gradually. Gains are modest on short dialogues (≤5 turns) but significant for longer contexts.

## Emotion-Anchored Contrastive Learning (EACL)
EACL reframes emotion classification as a **geometric learning problem** with **trainable emotion anchors** (one per class). Each utterance embedding is pulled toward its anchor and pushed from others, reshaping embedding geometry.

![Emotion anchors and embedding geometry](images/eacl-anchor.png)

**Figure 5.** Utterance embeddings and learned emotion anchors.

EACL sharpens similarity and angular-distance patterns, clarifying class boundaries.

![Similarity and angular distances before and after EACL](images/EACL-heatmap.png)

**Figure 6.** Similarity and angular distances before and after applying EACL.

## Knowledge-Enriched Transformer (KET)
While CMN, Transformer-XL, and EACL operate purely on text, real conversations often express emotions indirectly. KET injects **external knowledge**.

![Knowledge-Enriched Transformer architecture](images/KET_architecture.png)

**Figure 7.** Architecture of the Knowledge-Enriched Transformer (KET).

### What KET Adds
- **Commonsense knowledge (ConceptNet):** links words to events/situations (e.g. funeral -> sadness, celebration -> joy)
- **Affective signals (NRC-VAD):** valence, arousal, dominance scores
- **Knowledge-enriched representations:** injected before transformer layers so attention works over emotionally informed embeddings

### Why It Helps
- Recognizes implied emotions
- Improves subtle/minority classes (e.g. fear, disgust)
- Acts as an add-on for CMN, Transformer-XL, or EACL

## New Insights: Beyond Individual Architectures
Looking across CMN, Transformer-XL, and EACL, **context modeling** and **representation learning** cannot be treated separately. Failure modes cluster around borderline emotions (e.g. frustration vs. anger).

CMN handles short-range dependencies and identity cues but struggles with long build-ups. Transformer-XL extends context yet has less structured embeddings. EACL imposes clean anchor-based geometry but operates with more limited context.

![Complementary capabilities of CMN, Transformer-XL, and EACL](images/newinsight_flowchart.svg)

**Figure 8.** Each architecture contributes a distinct capability—speaker memory, long-range recurrence, and structured emotion space—that converges toward a hybrid design.

Taken together: **emotion in conversation is simultaneously a temporal and a geometric phenomenon.** Models need mechanisms for temporal evolution and a representation space where related emotions remain close but separable. This motivates a hybrid combining Transformer-XL–style recurrence with EACL's anchors.

![Embedding geometry](images/newinsight_compare.svg)

**Figure 9.** Emotion anchors reshape embedding geometry, separating borderline categories.

Our predicted evaluation of the Hybrid model shows a modest overall F1 lift, but with meaningful error shifts: reduced oscillation on long conversations and sharper boundaries between adjacent emotions such as **angry** and **frustrated**.

> **Future Addition**  
> Beyond these core components, external knowledge remains another complementary dimension. While not explicitly integrated into our Hybrid architecture, **KET-style knowledge enrichment** can be layered on top of any of the models we examined, providing commonsense grounding for emotions expressed implicitly (e.g. funeral -> sadness, celebration -> joy). In this sense, KET **does not compete** with our Hybrid model. Rather, it **illustrates a parallel axis of improvement-semantic grounding** that can further strengthen systems built on temporal modeling and geometric separation.

## Comparative Analysis
| Model | Strengths | Weaknesses | F1 |
| --- | --- | --- | --- |
| **CMN (2018)** | Local emotional cues; speaker-specific memory | Poor long-range context modeling | 56.5 |
| **Transformer-XL (2019)** | Stable long-context recurrence | Weak class separability near ambiguous emotions | 62.0 |
| **EACL (2024)** | Clear emotion clusters; structured representation space | Limited modeling of long-range conversational dynamics | 66.0 |
| **Hybrid** | Long-context recurrence *plus* anchor-based geometry | Higher complexity; harder to train and interpret ablations | **70.3\*** |

The Hybrid shows a small overall F1 gain (≈ +4.3 over EACL), concentrated on clearer separation of ambiguous emotions—especially *frustrated* vs. *angry*—and more stable predictions across long dialogues.

## Key Insights
1. **Different models capture different aspects of emotional context**  
   CMN focuses on speaker memory, Transformer-XL on long-range context, and EACL on cleaner emotion separation.
2. **Combining ideas is more powerful than relying on a single architecture**  
   Memory, long context, and structured emotion representations complement one another.
3. **External knowledge fills a gap that text alone cannot cover**  
   KET shows commonsense and affective knowledge helps interpret implicit or ambiguous emotions.

## GitHub Repository
https://github.com/GeorgiosIoannouCoder/ecem-csa

## Poster
https://docs.google.com/presentation/d/163czRoc6PsZhhk1dh0-AMqNHoTW67tLR24oUeSHu0_A

## References
- Poria, S., Hazarika, D., Majumder, N., & Mihalcea, R. (2018). **Conversational Memory Network (CMN)**. https://ww.sentic.net/conversational-memory-network.pdf
- Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**. https://arxiv.org/abs/1901.02860
- Zhang, T., Lin, S., Li, Y., & Wang, J. (2024). **Emotion-Anchored Contrastive Learning (EACL) for Conversational Emotion Recognition**. https://arxiv.org/abs/2403.20289
- Ye, X., Zhu, X., & Jiang, Y. (2019). **Knowledge-Enriched Transformer (KET)**. https://arxiv.org/abs/1909.10681
