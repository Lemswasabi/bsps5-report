X_train = np.reshape(X_train.values,(X_train.shape[0],32,20))
X_train, X_test, y_train, y_test = train_test_split(X_train,y)
