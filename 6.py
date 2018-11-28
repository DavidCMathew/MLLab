import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


msg = pd.read_csv('data/naive.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
y = msg.labelnum

xtrain, xtest, ytrain, ytest = train_test_split(X, y)

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)


print(count_vect.get_feature_names())

clf = MultinomialNB().fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

print('Accuracy metrics')
print('Accuracy of the classifer is', metrics.accuracy_score(ytest, predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, predicted))

print('Recall and Precison ')
print(metrics.recall_score(ytest, predicted))
print(metrics.precision_score(ytest, predicted))
'''
docs_new = ['I like this place', 'I like to work']
X_new_counts = count_vect.transform(docs_new)
predictednew = clf.predict(X_new_counts)
for doc, category in zip(docs_new, predictednew):
    print('%s->%s' % (doc, msg.labelnum[category]))
'''