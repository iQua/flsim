import logging
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time

class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id

    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]))

    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard

    # Server interactions
    def download(self, argv):
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv

    # Federated learning phases
    def set_data(self, data, config):
        # Extract from config
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        data = self.data
        if do_test:  # Partition for testset if applicable
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data

    # Set the speed for the link between server and client
    def set_link(self, config):
        # Set the Gaussian distribution for link speed in Kbytes
        self.speed_min = config.link.min
        self.speed_max = config.link.max
        self.speed_mean = random.uniform(self.speed_min, self.speed_max)
        self.speed_std = config.link.std

    def configure(self, config):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        model_path = self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent global model
        path = model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

        # Set the link speed and delay for the upcoming run
        model_size = os.path.getsize(path) / 1e3  # model size in Kbytes
        link_speed = random.normalvariate(self.speed_mean, self.speed_std)
        link_speed = max(min(link_speed, self.speed_max), self.speed_min)
        self.delay = model_size / link_speed  # upload delay in sec
        logging.info("client mean delay {} s".format(self.delay))

    def run(self):
        # Perform federated learning task
        {
            "train": self.train()
        }[self.task]

        #time.sleep(self.delay)

    def get_report(self):
        # Report results to server.
        return self.upload(self.report)

    # Machine learning tasks
    def train(self):
        import fl_model  # pylint: disable=import-error

        logging.info('Training on client #{}'.format(self.client_id))

        # Perform model training
        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        fl_model.train(self.model, trainloader,
                       self.optimizer, self.epochs)

        # Extract model weights and biases
        weights = fl_model.extract_weights(self.model)

        # Generate report for server
        self.report = Report(self)
        self.report.weights = weights

        # Perform model testing if applicable
        if self.do_test:
            testloader = fl_model.get_testloader(self.testset, 1000)
            self.report.accuracy = fl_model.test(self.model, testloader)

    def test(self):
        # Perform model testing
        raise NotImplementedError


class Report(object):
    """Federated learning client report."""

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
