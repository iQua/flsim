from server import Server
import numpy as np
import torch


class AccAvgServer(Server):
    """Federated learning server that performs accuracy weighted federated averaging."""

    # Federated learning phases
    def aggregation(self, reports):
        return self.accuracy_fed_avg(reports)

    # Report aggregation
    def accuracy_fed_avg(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract client accuracies
        accuracies = np.array([report.accuracy for report in reports])

        # Determine weighting based on accuracies
        factor = 8  # Exponentiation factor
        w = accuracies**factor / sum(accuracies**factor)

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            for j, (_, delta) in enumerate(update):
                # Use weighted average by magnetude of updates
                avg_update[j] += delta * w[i]

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    # Server operations
    def set_client_data(self, client):
        super().set_client_data(client)

        # Send each client a testing partition
        client.testset = client.download(self.loader.get_testset())
        client.do_test = True  # Tell client to perform testing
