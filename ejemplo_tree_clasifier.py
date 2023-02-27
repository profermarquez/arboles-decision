# https://sitiobigdata.com/2019/12/14/arbol-de-decision-en-machine-learning-parte-1/#
# https://www.ibm.com/es-es/topics/decision-trees
# https://es.wikipedia.org/wiki/Coeficiente_de_Gini

from sklearn import tree
import matplotlib.pyplot as plt

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

tree.plot_tree(clf)
plt.show()