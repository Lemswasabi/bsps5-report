lstm = Sequential()
lstm.add(LSTM(256, return_sequences=True,
                  input_shape=(X_train.shape[1],X_train.shape[2])))
lstm.add(Dropout(0.3))
lstm.add(LSTM(128))
lstm.add(Dropout(0.3))
lstm.add(Dense(10, activation='softmax'))
lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
historylstm = lstm.fit(X_train, y_train, validation_split=0.25, epochs=50,
                       batch_size=128, verbose=1)
test_loss, test_acc_lstm = lstm.evaluate(X_test, y_test)
