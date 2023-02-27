## Ejemplo 1
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

csvfile = """label,x,y
class1,10.0,8.04
class1,10.5,7.30
class2,8.3,5.5
class2,8.1,5.9
class3,3.5,3.5
class3,3.8,5.1
class1, 10.0, 5.5
class 2, 10, 6.3
class1, 7.9, 5.2
class3, 1.2, 5.9
class3, 10.0,0.3
class1, 7.2, 0.4
class2,5.0, 0.2
class1, 5.0, 7.0
class2, 10, 0.9
class3, 10, 8.9
class2, 5.3, 1.9
class1, 2.0, 9.8
class3, 3.0, 4.2
class1,12.0,9.04
class1,4.5,6.30
class2,6.3,3.5
class2,7.1,4.9
class3,8.5,2.5
class3,2.8,4.1
class1, 9.0, 4.5
class 2, 9.1, 9.3
class1, 8.2, 3.2
class3, 10.2, 1.9"""
df = pd.read_csv(StringIO(csvfile))
print(df)

from mlxtend.plotting import category_scatter  
fig = category_scatter(x='x', y='y', label_col='label',data=df, legend_loc='upper left')
fig.show()

## Ejemplo 2
from mlxtend.plotting import heatmap
import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')
df.columns = ['sample %d' % i for i in range(1, 15)]
df.head()

cols = ['sample 1', 'sample 5', 'sample 9', 'sample 12', 'sample 14']
cm = np.corrcoef(df[cols].values.T)
heatmap(cm, 
        column_names=cols, 
        row_names=cols,
        cmap = 'magma',
        figsize =(7.5, 7.5),
        cell_font_size = 20)
plt.show()