rnn = Sequential()
rnn.add(SimpleRNN(256, return_sequences=True,
                  input_shape=(X_train.shape[1],X_train.shape[2])))
rnn.add(Dropout(0.3))
rnn.add(SimpleRNN(128))
rnn.add(Dropout(0.3))
rnn.add(Dense(10, activation='softmax'))
rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

historyrnn = rnn.fit(X_train, y_train, validation_split=0.25, epochs=50,
                     batch_size=128, verbose=1)
test_loss, test_acc_rnn = rnn.evaluate(X_test, y_test)
