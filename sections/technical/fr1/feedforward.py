feedForward = Sequential()
feedForward.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
feedForward.add(Dropout(0.2))
feedForward.add(Dense(128, activation='relu'))
feedForward.add(Dropout(0.2))
feedForward.add(Dense(64, activation='relu'))
feedForward.add(Dropout(0.2))
feedForward.add(Dense(10, activation='softmax'))
feedForward.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

historyfeedForward = feedForward.fit(X_train, y_train, validation_split=0.25,
    epochs=60, batch_size=128, verbose=1)
test_loss, test_acc_ff = feedForward.evaluate(X_test, y_test)
