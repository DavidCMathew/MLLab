import pandas as pd

mush = pd.read_csv("data/flu.csv")
# the first column in dataset is class which is target variable
target = 'flu'
features = mush.columns[mush.columns != target]
classes = mush[target].unique()
test = mush.sample(frac=.3)
mush = mush.drop(test.index)

probs = {}
probcl = {}
for x in classes:
    tot = len(mushcl)
    mushcl = mush[mush[target] == x][features]
    probcl[x] = float(tot/len(mush))
    clsp = {}
    for col in mushcl.columns:
        colp = {}
        for val, cnt in mushcl[col].value_counts().iteritems():
            pr = cnt/tot
            colp[val] = pr
            clsp[col] = colp
        probs[x] = clsp


def probabs(x):
    probab = {}
    for cl in classes:
        pr = probcl[cl]
        for col, val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except:
                pr = 0
        probab[cl] = pr
    return probab


def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ''
    for cl, pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl


b = []
for i in mush.index:
    print(classify(mush.loc[i, features]), mush.loc[i, target])
    b.append(classify(mush.loc[i, features]) == mush.loc[i, target])
print(sum(b), "correct of", len(mush))
print("Accuracy:", sum(b)/len(mush))

# Test data
b = []
for i in test.index:
    print(classify(test.loc[i, features]), test.loc[i, target])
    b.append(classify(test.loc[i, features]) == test.loc[i, target])
print(sum(b), "correct of", len(test))
print("Accuracy:", sum(b)/len(test))
