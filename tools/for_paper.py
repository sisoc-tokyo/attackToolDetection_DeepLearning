import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import sys, shutil

from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras import utils

print("You have TensorFlow version", tf.__version__)

best_accuracy = 0

def learning(noise_rate, csv_file, epoch):
    #f.write('\n\n' + csv_file + '\nepoch: ' + str(epoch) + '\n')
    # learning data
    data = pd.read_csv(csv_file)
    token_data = 'nr' + str(noise_rate) + 'ep' + str(epoch) + '_' + 'token.pickle'
    encode_data = 'nr' + str(noise_rate) + 'ep' + str(epoch) + '_' + 'encode.pickle'
    model_data = 'nr' + str(noise_rate) + 'ep' + str(epoch) + '_' + 'dlls_model.h5'
    data = data.sample(frac=1)

    tag_num = data['tags'].nunique()
    data['tags'].value_counts()

    max_words  = 10000
    tokenizer = text.Tokenizer(num_words=max_words, char_level=False)

    max_len = 50
    tokenizer.fit_on_texts(data['data'])
    sequences = tokenizer.texts_to_sequences(data['data'])
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data_sec = pad_sequences(sequences, maxlen=max_len)

    import pickle
    # save the token data in the file
    with open(token_data, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Split data into train and test
    train_size = int(len(data_sec) * .8)
    print ("The number of train data: %d" % train_size)
    print ("The number of test data: %d" % (len(data_sec) - train_size))

    x_train = data_sec[:train_size]
    x_test = data_sec[train_size:]

    train_tags = data['tags'][:train_size]
    test_tags = data['tags'][train_size:]

    # Use sklearn utility to convert label strings to numbered index
    encoder = LabelEncoder()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    # save the encoder in the file
    with open(encode_data, 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # Build the model
    model = Sequential()
    model.add(Embedding(10000, 128, input_length=max_len))
    # lstm=LSTM(32)
    # print(lstm.units)
    optimizer = Adam()
    #optimizer = RMSprop()
    model.add(LSTM(32))
    model.add(Dense(tag_num, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=epoch,
                        verbose=1,
                        validation_data=(x_test, y_test))

    # Evaluate the accuracy of our trained model
    score = model.evaluate(x_test, y_test,
                           batch_size=32, verbose=1)
    print('Loss of evaluation:', score[0])
    print('Accuracy of evaluation:', score[1])

    #save model to the file
    model.save(model_data)

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    # loss
    def plot_history_loss(history):
        axL.plot(history.history['loss'],label="loss for training")
        axL.plot(history.history['val_loss'],label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')

    # acc
    def plot_history_acc(history):
        axR.plot(history.history['acc'],label="loss for training")
        axR.plot(history.history['val_acc'],label="loss for validation")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='lower right')

    plot_history_loss(history)
    plot_history_acc(history)
    fig.savefig('./' + 'nr' + str(noise_rate) + '_ep' + str(epoch) + '_' +  'history.png')
    plt.close()


def eval(noise_rate, epoch, count):
    path = 'eval.txt'
    with open(path, mode='a') as f:
        f.write('file_name: ' + file + ', epoch: ' + str(epoch) + ', n: ' + str(count) + '\n')
        model_name = 'nr' + str(noise_rate) + 'ep' + str(epoch) + '_' + 'dlls_model.h5'
        token_name = 'nr' + str(noise_rate) + 'ep' + str(epoch) + '_' + 'token.pickle'
        encoder_name = 'nr' + str(noise_rate) + 'ep' + str(epoch) + '_' + 'encode.pickle'
        model = load_model(model_name, compile=False)
        with open(token_name, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(encoder_name, 'rb') as handle:
            encoder = pickle.load(handle)
        data = pd.read_csv("00_test.csv")
        eval_data = data['data']
        eval_tag = data['tags']

        max_len = 50
        testdata_mat = tokenizer.texts_to_sequences(eval_data)
        data_sec = pad_sequences(testdata_mat, maxlen=max_len)
        prediction = model.predict(np.array(data_sec))

        fn = 0
        fp = 0
        tn = 0
        tp = 0

        for i in range(len(prediction)):
            predicted_label = encoder.classes_[np.argmax(prediction[i])]
            if predicted_label == eval_tag[i]:
                if eval_tag[i] == 'normal':
                    tn += 1
                else:
                    tp += 1
            else:
                if eval_tag[i] == 'normal':
                    fn += 1
                    f.write(str(i+1) + '\n')
                    f.write('True value: ' +str(eval_tag[i]) + '\n')
                    f.write('Predicted value: ' + str(predicted_label) + '\n')
                    #f.write(eval_data[i] + '\n')
                    f.write('\n')
                else:
                    fp += 1
                    f.write(str(i+1) + '\n')
                    f.write('True value: ' +str(eval_tag[i]) + '\n')
                    f.write('Predicted value: ' + str(predicted_label) + '\n')
                    #f.write(eval_data[i] + '\n')
                    f.write('\n')

        recall = tp / (tp + fp + 1e-09)
        precision = tp / (tp + fn + 1e-09)
        accuracy = (tp + tn) / (fn + fp + tn + tp + 1e-09)

        f.write('N: ' + str(i) + '\n' + 'Recall: ' + '{:.3f}'.format(recall) + ', Precision: ' + '{:.3f}'.format(precision) + ', Accuracy: ' + '{:.3f}'.format(accuracy) + '\n\n')
        print('N: ' + str(i) + '\n' + 'Recall: ' + '{:.3f}'.format(recall) + ' Precision: ' + '{:.3f}'.format(precision) + ' Accuracy: ' + '{:.3f}'.format(accuracy))

        # Copy if get better result.
        if accuracy >= best_accuracy:
            path = 'best_eval.txt'
            with open(path, mode='w') as bf:
                bf.write('file_name: ' + file + ', epoch: ' + str(epoch) + ', n: ' + str(count) + '\n')
                bf.write('N: ' + str(i) + '\n' + 'Recall: ' + '{:.3f}'.format(recall) + ', Precision: ' + '{:.3f}'.format(
                precision) + ', Accuracy: ' + '{:.3f}'.format(accuracy) + '\n\n')
                bf.close()
            # shutil.copy2(model_name, 'best_' + model_name)
            # shutil.copy2(token_name, 'best_' + token_name)
            # shutil.copy2(encoder_name, 'best_' + encoder_name)
        shutil.copy2(model_name, str(count) + model_name)
        shutil.copy2(token_name, str(count) + token_name)
        shutil.copy2(encoder_name, str(count) + encoder_name)

        # Stop program if perfect detection.
        #if recall > 0.99 and precision > 0.99:
            #sys.exit()

        f.write('\n')
        f.close()

file_list = ['00_train.csv']
epoch_list = [60]
for file in file_list:
    for epoch in epoch_list:
        count = 0
        for count in range(10):
            noise_rate = file.strip('train.csv')
            print(file + ', ' + str(epoch) + ', ' +str(count))
            learning(noise_rate, file, epoch)
            eval(noise_rate, epoch, count)
            count += 1