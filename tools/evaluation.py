import numpy as np
import pandas as pd
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def noise_eval(noise_rate, n):
    path = 'noise_eval/noise_eval.txt'
    with open(path, mode='a') as f:

        model = load_model('16nr00_ep60_dlls_model.h5', compile=False)
        with open('16nr00_ep60_token.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('16nr00_ep60_encode.pickle', 'rb') as handle:
            encoder = pickle.load(handle)

        noise_rate = "{0:02d}".format(noise_rate)
        #data = pd.read_csv('noise_eval/' + str(noise_rate) + '_test_' + str(n) + '.csv')
        #data = pd.read_csv('noise_eval/reduction' + str(noise_rate) + '_test_' + str(n) + '.csv')
        data = pd.read_csv('noise_eval/add_reduce' + str(noise_rate) + '_test_' + str(n) + '.csv')
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

        f.write('Noise_rate: ' + str(noise_rate) + '\n' + 'n: ' + str(n) + '\n' + 'Recall: ' + '{:.3f}'.format(recall) + ', Precision: ' + '{:.3f}'.format(precision) + ', Accuracy: ' + '{:.3f}'.format(accuracy) + '\n\n')
        print('Noise_rate: ' + str(noise_rate) + ' n: ' + str(n) + '\n' + 'Recall: ' + '{:.3f}'.format(recall) + ' Precision: ' + '{:.3f}'.format(precision) + ' Accuracy: ' + '{:.3f}'.format(accuracy))

    f.close()

noise_list = [5,10,15,20,25,30,35,40]
for noise_rate in noise_list:
    for n in range(100):
        noise_eval(noise_rate, n)
