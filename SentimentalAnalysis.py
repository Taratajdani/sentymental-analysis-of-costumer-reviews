import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# 1. Read the dataset
data = pd.read_csv('Amazon_DataSet.csv')

# 2. Preprocess the data
data['Label'] = data['Label'].map({'Positive': 1, 'Negative': 0})
X = data['Review']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tokenization and Stopword removal can be done using CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 3. Build the model
model = SVC()
model.fit(X_train_vectorized, y_train)

# 4. Test the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Visualization (optional)
plt.hist(y_pred)
plt.xlabel('Predicted Labels')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis Results')
plt.show()
