import numpy as np
import csv
import matplotlib.pyplot as plt

class Record(object):
    """Training records."""
    def __init__(self):
        self.t = []
        self.acc = []

    def append_acc_record(self, t, acc):
        self.t.append(t)
        self.acc.append(acc)

    def get_latest_t(self):
        return self.t[-1]

    def get_latest_acc(self):
        return self.acc[-1]

    def save_acc_record(self, filename):
        assert (len(self.t) == len(self.acc)), \
            "Length of time and acc records do not match!"

        t = np.expand_dims(np.array(self.t), axis=1)
        acc = np.expand_dims(np.array(self.acc), axis=1)
        rows = np.concatenate((t, acc), axis=1).tolist()

        fields = ['time', 'acc']
        with open(filename, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            for row in rows:
                write.writerow(row)

    def plot_acc_record(self, figname):
        assert (len(self.t) == len(self.acc)), \
            "Length of time and acc records do not match!"

        fig = plt.figure()
        plt.plot(self.t, self.acc, label='global acc')
        plt.xlabel('Time (s)')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(figname)
        plt.close(fig)