# Human Action Recognition on Static Images

---

## 📝 Introduction

Human Action Recognition (HAR) in computer vision is vital for applications such as surveillance, healthcare, and human-computer interaction. Traditional approaches depended on video-based inputs or handcrafted features, often struggling with complex actions. Recent advances leverage **static images** and **multimodal models** like [CLIP](https://github.com/mlfoundations/open_clip), which align textual and visual embeddings for robust action recognition.  
This project explores prompt engineering and Top-K accuracy to showcase how CLIP achieves **state-of-the-art performance** on static image action recognition, outperforming traditional CNNs.

---

## 🚀 Highlights

- **Static Image Action Recognition:** No video required; CLIP recognizes human actions from single images.
- **Multimodal Modeling:** Combines textual prompts and visual features via CLIP.
- **Prompt Engineering:** Custom textual prompts enhance action label discrimination.
- **Top-K Evaluation:** Assesses model ranking capabilities beyond Top-1 accuracy.
- **Visualization:** Self-attention maps reveal model interpretability.

---


## 🧪 Experiments

### Dataset

- [Human Action Recognition (HAR) Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset/)
- **15 classes**, >12,000 labeled images
- Images are organized in class-specific folders for streamlined training & evaluation


---

### Evaluation Metrics

- **Top-K Accuracy:**  
  - Evaluates if the true label appears in the top-K predictions (commonly K=5)
  - Especially useful when multiple actions are plausible for a single image
  - For test data (no GT): show top-3 model predictions

---

### Results

- **ResNet18:** Struggled with inter-class variance, limited generalization.
- **CLIP:** Achieved strong semantic alignment between text and image features. Prompt engineering further boosted recognition of fine-grained actions.

---

## 📊 Analysis & Visualization

- **CLIP** can overfit on training but still generalizes well (as observed in loss/accuracy plots).
- **No ground truth for test set:** Top-3 predictions shown for interpretability.
- **Attention maps** highlight the synergy between textual prompts and visual focus.

---