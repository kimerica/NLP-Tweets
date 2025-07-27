# Disaster Tweet Classification using NLP & Deep Learning

This project classifies tweets as either disaster-related (1) or not (0) using Natural Language Processing (NLP) techniques and deep learning models. The challenge lies in the informal, ambiguous, and noisy nature of real-world tweets.

---

## Dataset

- **Source:** Kaggle – [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- **Training Samples:** ~7,600 tweets
- **Test Samples:** ~3,200 tweets
- **Label:** `target` (1 = disaster, 0 = not disaster)

---

## Tools & Libraries

- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow`, `keras`, `torch`
- **Deep Learning Models:**  
  - Bidirectional LSTM  
  - GRU  
  - Simple RNN  
- **Vectorization:** TF-IDF, CountVectorizer

---

## Workflow

1. Load and explore the tweet dataset
2. Clean text using regular expressions (remove URLs, special characters, etc.)
3. Vectorize using TF-IDF or CountVectorizer
4. Build deep learning models with Keras (LSTM, GRU, RNN)
5. Train/validate models and monitor loss and accuracy
6. Predict on test data and evaluate results

---

## Evaluation

- Metrics used: **Accuracy**, **Loss**, and visual learning curves
- Applied **EarlyStopping** to prevent overfitting
- Compared model performance across LSTM, GRU, and RNN

> Bidirectional LSTM provided the best generalization across validation data.

---

## How to Run

1. Clone this repository
2. Download `train.csv` and `test.csv` from Kaggle
3. Install required packages: 'pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch'
4. Run the notebook: `disaster-tweet-classifier.ipynb`

---

## Author

**Erica Kim**  
Master’s in Data Science – University of Colorado Boulder  
[GitHub Profile](https://github.com/kimerica)
