import re
import pandas as pd
import codecs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = 'eval.txt'
with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
    lines = f.readlines()
    columns = ['noise_rate','epoch', 'Recall', 'Precision', 'Accuracy']
    df = pd.DataFrame([],columns=columns)

    n = 0
    for line in lines:
        nrp = re.compile('file_name: ([0-9][0-9])')
        nr = nrp.match(line)
        if nr:
            data_list = []
            nr = nr.group(1)
            data_list.append(nr)
        epp = re.compile('epoch: ([0-9][0-9][0-9])')
        ep = epp.search(line)
        if ep is None:
            epp = re.compile('epoch: ([0-9][0-9])')
            ep = epp.search(line)
        if ep:
            ep = int(ep.group(1))
            data_list.append(ep)

        rcp = re.compile('Recall: ([0-9]\.[0-9][0-9][0-9])')
        rc = rcp.search(line)
        if rc:
            rc = rc.group(1)
            data_list.append(rc)

        prp = re.compile('Precision: ([0-9]\.[0-9][0-9][0-9])')
        pr = prp.search(line)
        if pr:
            pr = pr.group(1)
            data_list.append(pr)

        acp = re.compile('Accuracy: ([0-9]\.[0-9][0-9][0-9])')
        ac = acp.search(line)
        if ac:
            ac = ac.group(1)
            data_list.append(ac)
            df.loc[n,:] = data_list
            n += 1

    df['Recall']=df['Recall'].astype(float)
    df['Precision'] = df['Precision'].astype(float)
    df['Accuracy'] = df['Accuracy'].astype(float)

    pv_recall = df.pivot_table(values='Recall',index='epoch',columns='noise_rate')
    pv_precision = df.pivot_table(values='Precision',index='epoch',columns='noise_rate')

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)

    pv_recall.plot(marker='o')

    plt.title('Recall transition', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.3)
    plt.tick_params(labelsize=7)
    plt.legend(loc='lower right', fontsize=12)
    plt.subplots_adjust(left=0.1, right=0.8)
    plt.tick_params(labelsize=12)

    plt.show()
    plt.savefig('Recall_transition.png')
