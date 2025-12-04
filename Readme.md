# KET Knowledge-Augmented Extension for Conversational Emotion Recognition

This module extends our team's baseline ERC models (CMN, Transformer-XL, EACL)
by injecting external knowledge into the encoder representations. Inspired by
the Knowledge-Enriched Transformer (KET), we incorporate two knowledge sources:

1. Commonsense knowledge from ConceptNet
2. Affective lexicon features from NRC-VAD (valence, arousal, dominance)

This extension does not modify teammates’ models internally. Instead, it wraps
their encoders and adds knowledge during training time, serving as an
optimization layer to improve emotional grounding and minority-class detection.

---

## 📁 File Overview

### **Core Files**
- `knowledge_module.py`  
  KET-style knowledge fusion block that merges token embeddings with knowledge
  features using a gated-add mechanism.

- `ket_wrapper.py`  
  Wrapper model that takes any base encoder (CMN / Transformer-XL / EACL) and
  outputs knowledge-augmented token states before classification.

- `train_ket_extension.py`  
  Training script demonstrating how to integrate the wrapper with knowledge
  features and emotion labels.

- `config_ket.yaml`  
  Configuration file specifying hyperparameters and paths.

---

## 🔧 Knowledge Resources

External knowledge is **not included** in this repository due to size and
license constraints. Instead, instructions for downloading and preprocessing
resources are provided in:

