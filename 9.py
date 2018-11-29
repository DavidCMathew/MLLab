from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

print("Class : number")
for i in range(len(iris_dataset.target_names)):
    print(iris_dataset.target_names[i], ':', i)

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"])
kn = KNeighborsClassifier(1)
kn.fit(X_train, y_train)

for i in range(len(X_test)):
    x_new = np.array([X_test[i]])
    prediction = kn.predict(x_new)
    print("Actual:{0} {1},Predicted:{2}{3}".format(y_test[i], iris_dataset["target_names"][y_test[i]], prediction, iris_dataset["target_names"][prediction]))

print("\nAccuracy: ", kn.score(X_test, y_test))
