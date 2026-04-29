# Multi-Label-Toxic-Comment-Detection-and-Automated-Content-Masking-Using-Hybrid-NLP-Techniques

An end-to-end NLP system that detects multiple types of toxicity in user comments and applies automated masking to reduce harm while preserving context.

---

## 🚀 Overview

This project builds a **multi-label toxic comment detection system** that goes beyond binary classification. Instead of simply labeling a comment as toxic or not, it identifies multiple toxicity categories simultaneously and applies **graduated masking techniques**.

The system combines **classical machine learning + hybrid NLP features**, making it computationally efficient and deployable without GPU.

---

## 🎯 Key Features

* ✅ Multi-label classification (6 toxicity types)
* ✅ Hybrid feature extraction (TF-IDF + Word2Vec)
* ✅ Comparison of 3 ML models
* ✅ Confidence-based risk scoring (SAFE → HIGH)
* ✅ Automated content masking (3 levels)
* ✅ Fully CPU-based implementation

---

## 🧩 Toxicity Categories

Each comment can belong to multiple labels:

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

---


2. **Feature Extraction**

   * TF-IDF (5000 features, bigrams)
   * Word2Vec (100-dim embeddings)

3. **Feature Fusion**

   * Concatenation → 5100-dim hybrid vector

4. **Classification**

   * Logistic Regression
   * Random Forest
   * Linear SVM (best performer)

5. **Risk Scoring**

   * SAFE / LOW / MEDIUM / HIGH

6. **Content Masking**

   * Full masking
   * Word-level masking
   * Partial masking

---

## ⚙️ Tech Stack

* Python 3.10
* Scikit-learn
* NLTK
* Gensim (Word2Vec)
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 📊 Dataset

* **Jigsaw Toxic Comment Classification Dataset**
* ~159k labeled comments
* Multi-label annotations
* Sample used: 20,000 comments

---

## 📈 Model Performance

| Model               | Performance |
| ------------------- | ----------- |
| Logistic Regression | Good        |
| Random Forest       | Moderate    |
| **Linear SVM**      | ⭐ Best      |

### Key Observations:

* Best results for: **toxic, obscene, insult**
* Weakest for: **threat (due to imbalance)**
* Hybrid features outperform single methods

---

## 🧪 Example Output

### 🔴 Toxic Comment

```
Input: You are a complete idiot and nobody likes you.

Prediction:
- toxic → MEDIUM
- insult → HIGH
- obscene → LOW

Masked Output:
You are a complete ***** and nobody likes you.
```

### 🟢 Clean Comment

```
Input: Great work on this project!

Prediction: SAFE
```

---

## 🛠️ Installation

```bash
pip install nltk scikit-learn pandas numpy matplotlib seaborn gensim kagglehub
```

---

## ▶️ How to Run

1. Clone the repo

```bash
git clone https://github.com/your-username/toxic-comment-detector.git
cd toxic-comment-detector
```

2. Open the notebook:

```bash
Enhanced_Toxic_Comment_Detector.ipynb
```

3. Run all cells

---

## 💡 Why This Approach Works

* **TF-IDF** → captures keyword importance
* **Word2Vec** → captures semantic meaning
* **Hybrid = Best of both worlds**

This avoids the heavy compute cost of transformer models while still achieving strong performance.

---

## ⚠️ Limitations

* Weak performance on rare labels (e.g., threat)
* No conversational context awareness
* Limited handling of slang / adversarial text
* Static toxic word dictionary

---

## 🔮 Future Improvements

* Fine-tune transformer models (BERT, RoBERTa)
* Add multilingual support
* Context-aware classification (conversation threads)
* Explainability using SHAP / LIME
* Dynamic slang detection

---

## 📌 Conclusion

This project demonstrates that **classical ML + smart feature engineering** can build an effective, scalable toxic comment detection system with real-world applicability.

---

## 👩‍💻 Author

**Astha Dakhinray**
BTech CSE (AI & ML)
