import numpy as np


class GreedyKCenter(object):
    def fit(self, points, k):
        centers = []
        centers_index = []
        # Initialize distances
        distances = [np.inf for u in points]
        # Initialize cluster labels
        labels = [np.inf for u in points]

        for cluster in range(k):
            # Let u be the point of P such that d[u] is maximum
            u_index = distances.index(max(distances))
            u = points[u_index]
            # u is the next cluster center
            centers.append(u)
            centers_index.append(u_index)

            # Update distance to nearest center
            for i, v in enumerate(points):
                distance_to_u = self.distance(u, v)  # Calculate from v to u
                if distance_to_u < distances[i]:
                    distances[i] = distance_to_u
                    labels[i] = cluster

            # Update the bottleneck distance
            max_distance = max(distances)

        # Return centers, labels, max delta, labels
        self.centers = centers
        self.centers_index = centers_index
        self.max_distance = max_distance
        self.labels = labels

    @staticmethod
    def distance(u, v):
        displacement = u - v
        return np.sqrt(displacement.dot(displacement))
