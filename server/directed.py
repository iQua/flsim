import logging
from server import Server
import numpy as np
from threading import Thread


class DirectedServer(Server):
    """Federated learning server that uses profiles to direct during selection."""

    # Run federated learning
    def run(self):
        # Perform profiling on all clients
        self.profiling()

        # Continue federated learning
        super().run()

    # Federated learning phases
    def selection(self):
        import fl_model  # pylint: disable=import-error

        clients = self.clients
        clients_per_round = self.config.clients.per_round
        profiles = self.profiles
        w_previous = self.w_previous

        # Extract directors from profiles
        directors = [d for _, d in profiles]

        # Extract most recent model weights
        w_current = self.flatten_weights(fl_model.extract_weights(self.model))
        model_direction = w_current - w_previous
        # Normalize model direction
        model_direction = model_direction / \
            np.sqrt(np.dot(model_direction, model_direction))

        # Update previous model weights
        self.w_previous = w_current

        # Generate client director scores (closer direction is better)
        scores = [np.dot(director, model_direction) for director in directors]
        # Apply punishment for repeatedly selected clients
        p = self.punishment
        scores = [x * (0.9)**p[i] for i, x in enumerate(scores)]

        # Select clients with highest scores
        sample_clients_index = []
        for _ in range(clients_per_round):
            top_score_index = scores.index(max(scores))
            sample_clients_index.append(top_score_index)
            # Overwrite to avoid reselection
            scores[top_score_index] = min(scores) - 1

        # Extract selected sample clients
        sample_clients = [clients[i] for i in sample_clients_index]

        # Update punishment factors
        self.punishment = [
            p[i] + 1 if i in sample_clients_index else 0 for i in range(len(clients))]

        return sample_clients

    def profiling(self):
        import fl_model  # pylint: disable=import-error

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

        # Extract initial model weights
        w0 = self.flatten_weights(fl_model.extract_weights(self.model))

        # Save as initial previous model weights
        self.w_previous = w0.copy()

        # Update initial model using results of profiling
        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Calculate direction vectors (directors)
        directors = [(w - w0) for w in weights]
        # Normalize directors to unit length
        directors = [d / np.sqrt(np.dot(d, d)) for d in directors]

        # Initialize punishment factors
        self.punishment = [0 for _ in range(len(clients))]

        # Use directors for client profiles
        self.profiles = [(client, directors[i])
                         for i, client in enumerate(clients)]
        return self.profiles
