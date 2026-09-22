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

# ----------------------------------第一步 读取文件--------------------------------
with open('train2.csv', 'r', encoding='UTF-8') as f:
    reader = csv.DictReader(f)
    fake_labels = []
    real_labels = []
    content = []
    for row in reader:
        fake_labels.append(row['fake_label'])  # 0-真 1-假
        real_labels.append(row['real_label'])
        content.append(row['content'])

print(fake_labels[:5])
print(real_labels[:5])
print(content[:5])

# ----------------------------------第二步 数据预处理--------------------------------
# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(min_df=5)

# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(content))
for n in tfidf[:5]:
    print(n)
print(type(tfidf))

# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names_out()
for n in word[:30]:
    print(n)
print("word count:", len(word))

# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
# X = tfidf.toarray()
X = coo_matrix(tfidf, dtype=np.float32).toarray()  # 稀疏矩阵 注意float
print(X.shape)
print(X[:10])

# ----------------------------------第三步 数据划分--------------------------------
# 使用 train_test_split 分割 X y 列表
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    fake_labels,
                                                    test_size=0.3,
                                                    random_state=1)

# --------------------------------第四步 classification--------------------------------
# 逻辑回归分类方法模型

# LR = LogisticRegression(solver='liblinear')
# LR.fit(X_train, y_train)
# print('Accuracy:{}'.format(LR.score(X_test, y_test)))
# pre = LR.predict(X_test)
# print("classification")
# print(len(pre), len(y_test))
# print(classification_report(y_test, pre))

#最近邻算法

erros = []
for i in range(1,14):        #150开根号
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    erros.append(1-accuracy)
pre = knn.predict(X_test)
import matplotlib.pyplot as plt
plt.plot(np.arange(1,14),erros)
plt.show()

# knn = neighbors.KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train, y_train)
# print('Accuracy:{}'.format(knn.score(X_test, y_test)))
# pre = knn.predict(X_test)
# print("KNN")
# print(classification_report(y_test, pre))
# print("\n")

# # SVM分类方法模型
# SVM = svm.LinearSVC() #支持向量机分类器LinearSVC
# SVM.fit(X_train, y_train)
# print('Accuracy:{}'.format(SVM.score(X_test, y_test)))
# pre = SVM.predict(X_test)
# print("Support Vector Machines Classification")
# print(len(pre), len(y_test))
# print(classification_report(y_test, pre))
# print("\n")

# ----------------------------------第五步 评价结果--------------------------------
def classification_pj(name, y_test, pre):
    print("evaluation:", name)

    # 正确率 Precision = 正确识别的个体总数 /  识别出的个体总数
    # 召回率 Recall = 正确识别的个体总数 /  测试集中存在的个体总数
    # F值 F-measure = 正确率 * 召回率 * 2 / (正确率 + 召回率)

    YC_B, YC_G = 0, 0  # 预测 fake real
    ZQ_B, ZQ_G = 0, 0  # 正确
    CZ_B, CZ_G = 0, 0  # 存在

    # 0-real 1-fake 同时计算防止类标变化
    i = 0
    while i < len(pre):
        z = int(y_test[i])  # real
        y = int(pre[i])  # predict

        if z == 0:
            CZ_G += 1
        else:
            CZ_B += 1

        if y == 0:
            YC_G += 1
        else:
            YC_B += 1

        if z == y and z == 0 and y == 0:
            ZQ_G += 1
        elif z == y and z == 1 and y == 1:
            ZQ_B += 1
        i = i + 1

    print(ZQ_B, ZQ_G, YC_B, YC_G, CZ_B, CZ_G)
    print("")

    # 结果输出
    P_G = ZQ_G * 1.0 / YC_G
    P_B = ZQ_B * 1.0 / YC_B
    print("Precision Good 0:", P_G)
    print("Precision Bad 1:", P_B)

    R_G = ZQ_G * 1.0 / CZ_G
    R_B = ZQ_B * 1.0 / CZ_B
    print("Recall Good 0:", R_G)
    print("Recall Bad 1:", R_B)

    F_G = 2 * P_G * R_G / (P_G + R_G)
    F_B = 2 * P_B * R_B / (P_B + R_B)
    print("F-measure Good 0:", F_G)
    print("F-measure Bad 1:", F_B)


# 函数调用

classification_pj("KNeighborsClassifier", y_test, pre)

# classification_pj("LogisticRegression", y_test, pre)

# classification_pj("SVM", y_test, pre)




# # 设置需要搜索的K值，'n_neightbors'是sklearn中KNN的参数
# parameters={'n_neightbors':[1,3,5,7,9,11,13,15]}
# knn = KNeighborsClassifier()
#
# # 通过GridSearchCV来搜索最好的K值。这个模块的内部其实就是对每一个K值进行评估
# clf=GridSearchCV(knn,parameters,cv=5)  #5折
# clf.fit(X_train, y_train)
# print("最终最佳准确率：%.2f"%clf.best_score_,"最终的最佳K值",clf.best_params_)

#最近邻算法
# knn = neighbors.KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train, y_train)
# print('Accuracy:{}'.format(knn.score(X_test, y_test)))
# pre = knn.predict(X_test)
# print("KNN")
# print(classification_report(y_test, pre))
# classification_pj("KNeighbors", y_test, pre)
# print("\n")
#
# # SVM分类方法模型
# SVM = svm.LinearSVC() #支持向量机分类器LinearSVC
# SVM.fit(X_train, y_train)
# print('Accuracy:{}'.format(SVM.score(X_test, y_test)))
# pre = SVM.predict(X_test)
# print("Support Vector Machines Classification")
# print(len(pre), len(y_test))
# print(classification_report(y_test, pre))
# classification_pj("LinearSVC", y_test, pre)
# print("\n")