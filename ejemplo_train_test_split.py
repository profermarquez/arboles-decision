# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)
print("imprimo X ")
print(X)
print("imprimo una lista de y ")
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("Imprimo X_train")
print(X_train)
print("Imprimo X_test")
print(X_test)
print("Imprimo y_train")
print(y_train)
print("Imprimo y_test")
print(y_test)