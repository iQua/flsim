import client
import load_data
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
import utils.dists as dists  # pylint: disable=no-name-in-module


class Server(object):
    """Basic federated learning server."""

    def __init__(self, config):
        self.config = config

    # Set up server
    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated server
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)

    def load_data(self):
        import fl_model  # pylint: disable=import-error

        # Extract config for loaders
        config = self.config

        # Set up data generator
        generator = fl_model.Generator()

        # Generate data
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        labels = generator.labels

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.debug('Labels ({}): {}'.format(
            len(labels), labels))

        # Set up data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator)
        }[self.config.loader]

        logging.info('Loader: {}, IID: {}'.format(
            self.config.loader, self.config.data.IID))

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, model_path)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

        # Make simulated clients
        clients = []
        for client_id in range(num_clients):

            # Create new client
            new_client = client.Client(client_id)

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]

                    # Assign preference, bias config
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard

                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        if loader == 'bias':
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loading == 'static':
            if loader == 'shard':  # Create data shards
                self.loader.create_shards()

            # Send data partition to all clients
            [self.set_client_data(client) for client in clients]

        self.clients = clients

    # Run federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the federated learning round
            accuracy = self.round()

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def round(self):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_clients = self.selection()

        # Configure sample clients
        self.configuration(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(sample_clients)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy

    # Federated learning phases

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        sample_clients = [client for client in random.sample(
            self.clients, clients_per_round)]

        return sample_clients

    def configuration(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            client.configure(config)

    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def aggregation(self, reports):
        return self.federated_averaging(reports)

    # Report aggregation
    def extract_client_updates(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def accuracy_averaging(self, reports):
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client):
        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

        # Extract data partition for client
        if loader == 'basic':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to client
        client.set_data(data, self.config)

    def save_model(self, model, path):
        path += '/global'
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model  # pylint: disable=import-error

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))
