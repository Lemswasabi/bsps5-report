gru = Sequential()
gru.add(GRU(256, return_sequences=True,
                  input_shape=(X_train.shape[1],X_train.shape[2])))
feedForward.add(Dropout(0.3))
gru.add(GRU(128))
feedForward.add(Dropout(0.3))
gru.add(Dense(10, activation='softmax'))
gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

historygru = gru.fit(X_train, y_train, validation_split=0.25, epochs=50,
                     batch_size=128, verbose=1)
test_loss, test_acc_gru = gru.evaluate(X_test, y_test)
