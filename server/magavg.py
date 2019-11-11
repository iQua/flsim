from server import Server
import numpy as np
import torch


class MagAvgServer(Server):
    """Federated learning server that performs magnetude weighted federated averaging."""

    # Federated learning phases
    def aggregation(self, reports):
        return self.magnetude_fed_avg(reports)

    # Report aggregation
    def magnetude_fed_avg(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract update magnetudes
        magnetudes = []
        for update in updates:
            magnetude = 0
            for _, weight in update:
                magnetude += weight.norm() ** 2
            magnetudes.append(np.sqrt(magnetude))

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            for j, (_, delta) in enumerate(update):
                # Use weighted average by magnetude of updates
                avg_update[j] += delta * (magnetudes[i] / sum(magnetudes))

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights
