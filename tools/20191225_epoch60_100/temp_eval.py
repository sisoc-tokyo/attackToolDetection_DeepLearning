import numpy as np
import pandas as pd
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def noise_eval(n):
    path = 'epoch60/epoch_60_eval.txt'
    with open(path, mode='a') as f:
        
        model = load_model('epoch60/' + str(n) + 'nr00_ep60_dlls_model.h5', compile=False)
        with open('epoch60/' + str(n) + 'nr00_ep60_token.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('epoch60/' + str(n) + 'nr00_ep60_encode.pickle', 'rb') as handle:
            encoder = pickle.load(handle)

        data = pd.read_csv('00_test.csv')
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

        f.write('Noise_rate: 60 n: ' + str(n) + '\n' + 'Recall: ' + '{:.3f}'.format(recall) + ' Precision: ' + '{:.3f}'.format(precision) + ' Accuracy: ' + '{:.3f}'.format(accuracy))
        f.write('\n-------------------------')
        f.write('\n\n')

for n in range(100):
    noise_eval(n)