import logging
import random
from server import Server
from sklearn.cluster import KMeans
from threading import Thread
import utils.dists as dists  # pylint: disable=no-name-in-module


class KMeansServer(Server):
    """Federated learning server that performs KMeans profiling during selection."""

    # Run federated learning
    def run(self):
        # Perform profiling on all clients
        self.profile_clients()

        # Continue federated learning
        super().run()

    # Federated learning phases
    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round
        cluster_labels = self.clients.keys()

        # Generate uniform distribution for selecting clients
        dist = dists.uniform(clients_per_round, len(cluster_labels))

        # Select clients from KMeans clusters
        sample_clients = []
        for i, cluster in enumerate(cluster_labels):
            # Select clients according to distribution
            if len(self.clients[cluster]) >= dist[i]:
                k = dist[i]
            else:  # If not enough clients in cluster, use all avaliable
                k = len(self.clients[cluster])

            sample_clients.extend(random.sample(
                self.clients[cluster], k))

         # Shuffle selected sample clients
        random.shuffle(sample_clients)

        return sample_clients

    # Output model weights
    def model_weights(self, clients):
        # Configure clients to train on local data
        self.configuration(clients)

        # Train on local data for profiling purposes
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports = self.reporting(clients)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        return [self.flatten_weights(weight) for weight in weights]

    def prefs_to_weights(self):
        prefs = [client.pref for client in self.clients]
        return list(zip(prefs, self.model_weights(self.clients)))

    def profiling(self, clients):
        # Perform clustering

        weight_vecs = self.model_weights(clients)

        # Use the number of clusters as there are labels
        n_clusters = len(self.loader.labels)

        logging.info('KMeans: {} clients, {} clusters'.format(
            len(weight_vecs), n_clusters))
        kmeans = KMeans(  # Use KMeans clustering algorithm
            n_clusters=n_clusters).fit(weight_vecs)

        return kmeans.labels_

    # Server operations
    def profile_clients(self):
        # Perform profiling on all clients
        kmeans = self.profiling(self.clients)

        # Group clients by profile
        grouped_clients = {cluster: [] for cluster in
                           range(len(self.loader.labels))}
        for i, client in enumerate(self.clients):
            grouped_clients[kmeans[i]].append(client)

        self.clients = grouped_clients  # Replace linear client list with dict

    def add_client(self):
        # Add a new client to the server
        raise NotImplementedError
