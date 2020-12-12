import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('largetrainingset.csv')
y = df['label']
X = df.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y)
