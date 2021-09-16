import numpy as np
import csv
import matplotlib.pyplot as plt

class Record(object):
    """Accuracy records."""
    def __init__(self):
        self.t = []
        self.acc = []
        self.throughput = []
        self.alpha = 0.1
        self.last_acc = 0

    def append_record(self, t, acc, throughput):
        self.t.append(t)
        self.throughput.append(throughput)
        if len(self.acc) == 0:
            self.acc.append(acc)
        else:
            self.acc.append((1 - self.alpha) * self.last_acc + \
                            self.alpha * acc)
        self.last_acc = self.acc[-1]

    def get_latest_t(self):
        return self.t[-1]

    def get_latest_acc(self):
        return self.acc[-1]

    def save_record(self, filename):
        assert (len(self.t) == len(self.acc)), \
            "Length of time and acc records do not match! t {} acc {}".format(
                len(self.t), len(self.acc)
            )
        assert (len(self.t) == len(self.throughput)), \
            "Length of time and throughput records do not match! t {} throughput {}".format(
                len(self.t), len(self.throughput)
            )

        t = np.expand_dims(np.array(self.t), axis=1)
        acc = np.expand_dims(np.array(self.acc), axis=1)
        throughput = np.expand_dims(np.array(self.throughput), axis=1)
        rows = np.concatenate((t, acc, throughput), axis=1).tolist()

        fields = ['time', 'acc', 'throughput']
        with open(filename, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            for row in rows:
                write.writerow(row)

    def plot_record(self, figname):
        assert (len(self.t) == len(self.acc)), \
            "Length of time and acc records do not match!"

        fig = plt.figure(figsize=(6, 8))
        plt.subplot(211)
        plt.plot(self.t, self.acc, label='global acc')
        plt.xlabel('Time (s)')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.subplot(212)
        plt.plot(self.t, self.throughput, label='throughput')
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (kB/s)')
        plt.legend()
        plt.savefig(figname)
        plt.close(fig)

class Profile(object):
    """Clients' loss and delay profile"""
    def __init__(self, num_clients):
        self.loss = [-1] * num_clients
        self.delay = [-1] * num_clients
        self.alpha = 0.1

    def update(self, client_idx, loss, delay):
        if self.loss[client_idx] > 0:
            # Not the first profile
            self.loss[client_idx] = loss
            self.delay[client_idx] = (1 - self.alpha) * self.delay[client_idx] + \
                self.alpha * delay
        else:
            self.loss[client_idx] = loss
            self.delay[client_idx] = delay

    def plot(self, figname):
        fig = plt.figure()
        plt.scatter(self.loss, self.delay, s=10)
        plt.xlabel('Loss')
        plt.xlim(left=.0)
        plt.ylabel('Delay (s)')
        plt.ylim(bottom=.0)
        plt.savefig(figname)
        plt.close(fig)