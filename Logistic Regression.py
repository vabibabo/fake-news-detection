import csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from scipy.sparse import coo_matrix
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hinge_loss
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# ----------------------------------Step 1 Read the file--------------------------------
with open('train2.csv', 'r', encoding='UTF-8') as f:
    reader = csv.DictReader(f)
    fake_labels = []
    real_labels = []
    content = []
    for row in reader:
        fake_labels.append(row['fake_label'])  # 0-real 1-fake
        real_labels.append(row['real_label'])
        content.append(row['content'])

print(fake_labels[:5])
print(real_labels[:5])
print(content[:5])

# ----------------------------------Step 2 Data pre-processing--------------------------------
# Convert words in the text into a word frequency matrix Element a[i][j] of the matrix represents the word frequency of word j under class i text
vectorizer = CountVectorizer(min_df=5)

# This counts the tf-idf weights of each word
transformer = TfidfTransformer()

# The first fit_transform is to calculate the tf-idf and the second fit_transform is to convert the text into a word frequency matrix
tfidf = transformer.fit_transform(vectorizer.fit_transform(content))
for n in tfidf[:5]:
    print(n)
print(type(tfidf))

# Get all words in the bag-of-words model
word = vectorizer.get_feature_names_out()
for n in word[:30]:
    print(n)
print("word count:", len(word))

# The tf-idf matrix is extracted and the element w[i][j] represents the tf-idf weight of word j in class i text
# X = tfidf.toarray()
X = coo_matrix(tfidf, dtype=np.float32).toarray()  # 稀疏矩阵 注意float
print(X.shape)
print(X[:10])

# ----------------------------------Step 3 Data segmentation--------------------------------
# Splitting the X y list using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    fake_labels,
                                                    test_size=0.5,
                                                    random_state=1)

# --------------------------------Step 4 classification--------------------------------


# Logistic Regression

LR = LogisticRegression(solver='liblinear')
LR.fit(X_train, y_train)
print('Accuracy:{}'.format(LR.score(X_test, y_test)))
pre = LR.predict(X_test)
print("classification")
print(len(pre), len(y_test))
print(classification_report(y_test, pre))


# ----------------------------------Step 5 Evaluation results--------------------------------
def classification_pj(name, y_test, pre):
    print("evaluation:", name)

    # Precision = Total number of correctly identified individuals / Total number of individuals identified
    # Recall = Total number of correctly identified individuals / Total number of individuals present in the test set
    # F-measure = Precision * Recall * 2 / (Precision + Recall)

    YC_F, YC_R = 0, 0  # Predict fake real
    ZQ_F, ZQ_R = 0, 0  # real
    CZ_F, CZ_R = 0, 0  # Existence

    # 0-real 1-fake Simultaneous calculations to prevent class label changes
    i = 0
    while i < len(pre):
        z = int(y_test[i])  # real
        y = int(pre[i])  # predict

        if z == 0:
            CZ_R += 1
        else:
            CZ_F += 1

        if y == 0:
            YC_R += 1
        else:
            YC_F += 1

        if z == y and z == 0 and y == 0:
            ZQ_R += 1
        elif z == y and z == 1 and y == 1:
            ZQ_F += 1
        i = i + 1

    print(ZQ_F, ZQ_R, YC_F, YC_R, CZ_F, CZ_R)
    print("")

    # result putput
    P_R = ZQ_R * 1.0 / YC_R
    P_F = ZQ_F * 1.0 / YC_F
    print("Precision Real 0:", P_R)
    print("Precision Fake 1:", P_F)

    R_R = ZQ_R * 1.0 / CZ_R
    R_F = ZQ_F * 1.0 / CZ_F
    print("Recall Real 0:", R_R)
    print("Recall Fake 1:", R_F)

    F_R = 2 * P_R * R_R / (P_R + R_R)
    F_F = 2 * P_F * R_F / (P_F + R_F)
    print("F-measure Real 0:", F_R)
    print("F-measure Fake 1:", F_F)


# Function calls

classification_pj("LogisticRegression", y_test, pre)


