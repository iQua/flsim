import argparse
import client
import config
import logging
import os
import pickle
from sklearn.decomposition import PCA
import server


# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=logging.INFO, datefmt='%H:%M:%S')

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./template.json',
                    help='Configuration file for server.')
parser.add_argument('-o', '--output', type=str, default='./output.pkl',
                    help='Output pickle file')

args = parser.parse_args()


def main():
    """Extract PCA vectors from FL clients."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = server.KMeansServer(fl_config)
    fl_server.boot()

    # Run client profiling
    fl_server.profile_clients()

    # Extract clients, reports, weights
    clients = [client for client in group for group in [
        fl_server.clients[profile] for profile in fl_server.clients.keys()]]
    reports = [client.get_report() for client in clients]
    weights = [report.weights for report in reports]

    # Flatten weights
    def flatten_weights(weights):
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten())
        return weight_vecs

    logging.info('Flattening weights...')
    weight_vecs = [flatten_weights(weight) for weight in weights]

    # Perform PCA on weight vectors
    logging.info('Assembling output...')
    output = [(clients[i].client_id, clients[i].pref, weight) for i, weight in enumerate(weight_vecs)]
    logging.into('Writing output to binary...')
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)

    logging.info('Done!')

if __name__ == "__main__":
    main()
