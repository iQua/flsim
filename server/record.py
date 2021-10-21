import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        self.loss = np.repeat(-1., num_clients)
        self.delay = np.repeat(-1., num_clients)
        self.primary_label = np.repeat(-1., num_clients)
        self.alpha = 0.1
        self.weights = [[]] * num_clients

    def set_primary_label(self, pref_str):
        """
        Note, pref is a list of string labels like '3 - three'
        We need to convert the list of string labels to integers
        """
        pref_int = [int(s.split('-')[0].strip()) for s in pref_str]
        self.primary_label = np.array(pref_int)

    def update(self, client_idx, loss, delay, flatten_weights):
        if self.loss[client_idx] > 0:
            # Not the first profile
            self.delay[client_idx] = (1 - self.alpha) * self.delay[client_idx] + \
                self.alpha * delay
        else:
            self.delay[client_idx] = delay
        self.loss[client_idx] = loss
        self.weights[client_idx] = flatten_weights

    def plot(self, T, path):
        """
        Plot the up-to-date profiles, including loss-delay distribution,
        and 2D PCA plots of weights
        Args:
            T: current time in secs
        """
        def get_cmap(n, name='hsv'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n + 1)

        fig = plt.figure()
        cmap = get_cmap(len(set(self.primary_label.tolist())))
        color_ind = 0
        for l in set(self.primary_label.tolist()):
            mask = (self.primary_label == l)
            plt.scatter(x=self.loss[mask], y=self.delay[mask], s=10,
                        color=cmap(color_ind), label=str(l))
            color_ind += 1
        plt.legend()
        plt.xlabel('Loss')
        plt.xlim(left=.0)
        plt.ylabel('Delay (s)')
        plt.ylim(bottom=.0)
        plt.savefig(path + '/ld_{}.png'.format(T))
        plt.close(fig)

        w_array, l_list = [], []
        for i in range(len(self.weights)):
            if len(self.weights[i]) > 0:  # weight is not empty
                w_array.append(self.weights[i])
                l_list.append(self.primary_label[i])
        w_array, l_array = np.array(w_array), np.array(l_list)
        w_array = StandardScaler().fit_transform(w_array)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(w_array)

        fig = plt.figure()
        cmap = get_cmap(len(list(set(l_list))))
        color_ind = 0
        for l in set(l_list):
            mask = (l_array == l)
            plt.scatter(x=pc[mask, 0], y=pc[mask, 1], alpha=0.8, s=20,
                       color=cmap(color_ind), label=str(l))
            color_ind += 1
        plt.legend()
        plt.title('PCA transform of weights profile')
        plt.savefig(path + '/pca_{}.png'.format(T))
        plt.close(fig)

