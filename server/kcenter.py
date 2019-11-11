import logging
import random
from server import Server
from threading import Thread
from utils.kcenter import GreedyKCenter  # pylint: disable=no-name-in-module


class KCenterServer(Server):
    """Federated learning server that performs KCenter profiling during selection."""

    # Run federated learning
    def run(self):
        # Perform profiling on all clients
        self.profiling()

        # Designate space for storing used client profiles
        self.used_profiles = []

        # Continue federated learning
        super().run()

    # Federated learning phases
    def selection(self):
        # Select devices to participate in round

        profiles = self.profiles
        k = self.config.clients.per_round

        if len(profiles) < k:  # Reuse clients when needed
            logging.warning('Not enough unused clients')
            logging.warning('Dumping clients for reuse')
            self.profiles.extend(self.used_profiles)
            self.used_profiles = []

        # Shuffle profiles
        random.shuffle(profiles)

        # Cluster clients based on profile weights
        weights = [weight for _, weight in profiles]
        KCenter = GreedyKCenter()
        KCenter.fit(weights, k)

        logging.info('KCenter: {} clients, {} centers'.format(
            len(profiles), k))

        # Select clients marked as cluster centers
        centers_index = KCenter.centers_index
        sample_profiles = [profiles[i] for i in centers_index]
        sample_clients = [client for client, _ in sample_profiles]

        # Mark sample profiles as used
        self.used_profiles.extend(sample_profiles)
        for i in sorted(centers_index, reverse=True):
            del self.profiles[i]

        return sample_clients

    def profiling(self):
        # Use all clients for profiling
        clients = self.clients

        # Configure clients for training
        self.configuration(clients)

        # Train on clients to generate profile weights
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports = self.reporting(clients)

        # Extract weights from reports
        weights = [report.weights for report in reports]
        weights = [self.flatten_weights(weight) for weight in weights]

        # Use weights for client profiles
        self.profiles = [(client, weights[i])
                         for i, client in enumerate(clients)]
        return self.profiles
