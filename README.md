# 📱 SMS Spam Classification

A machine learning project that classifies SMS messages as spam or ham (non-spam) using Natural Language Processing and Naive Bayes classification, achieving **96.63% accuracy**.

## 🎯 Project Overview
This project implements a text classification system that can automatically detect spam SMS messages. Using the UCI SMS Spam Collection Dataset, we've built a model that can effectively distinguish between legitimate messages (ham) and spam messages.

## 📊 Performance Metrics
- Accuracy: 96.63%
- F1-Score: 0.865
- Precision for Spam: 1.00
- Recall for Spam: 0.76

### Confusion Matrix
```
[[1196    0]
 [  47  150]]
```

## 🔧 Technical Details

### Dataset
- Total messages: 5,572
- Spam messages: 747 (13.4%)
- Ham messages: 4,825 (86.6%)
- Training/Test split: 75/25

### Implementation Features
- TF-IDF Vectorization for text feature extraction
- Multinomial Naive Bayes classifier
- Text preprocessing including:
  - Lowercase conversion
  - Stop words removal
  - English language optimization

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- scikit-learn
- numpy
- kaggle

### Dataset Setup
1. Install Kaggle package:
```bash
pip install kaggle
```

2. Configure Kaggle credentials:
```bash
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. Download the dataset:
```bash
kaggle datasets download -d uciml/sms-spam-collection-dataset
```

### Running the Project
1. Load and preprocess the data:
```python
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.rename(columns={'v1':'class_label', 'v2':'message'}, inplace=True)
```

2. Train the model:
```python
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
features_train_transformed = vectorizer.fit_transform(x_train)
classifier = MultinomialNB()
classifier.fit(features_train_transformed, y_train)
```

## 📈 Results Breakdown
- Perfect precision (1.00) for spam detection
- High recall (0.76) for spam messages
- Excellent overall accuracy (96.63%)
- Strong performance on imbalanced dataset

## 🔍 Model Evaluation
```
              precision    recall  f1-score   support
           0       0.96      1.00      0.98      1196
           1       1.00      0.76      0.86       197
```

## 📝 Future Improvements
- Experiment with other classification algorithms
- Implement cross-validation
- Add more features from message metadata
- Try different text preprocessing techniques
- Collect more spam examples to balance the dataset

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- UCI Machine Learning Repository for the SMS Spam Collection Dataset
- Kaggle for dataset hosting
- scikit-learn team for the machine learning tools

---
Created with 💌 and ML
