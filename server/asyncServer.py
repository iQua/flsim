import logging
import pickle
import random
from threading import Thread
import torch
from queue import PriorityQueue
import os
from server import Server
from .record import Record, Profile


class Group(object):
    """Basic async group."""
    def __init__(self, client_list):
        self.clients = client_list

    def set_download_time(self, download_time):
        self.download_time = download_time

    def set_aggregate_time(self):
        """Only run after client configuration"""
        assert (len(self.clients) > 0), "Empty clients in group init!"
        self.delay = max([c.delay for c in self.clients])
        self.aggregate_time = self.download_time + self.delay

        # Get average throughput contributed by this group
        assert (self.clients[0].model_size > 0), "Zero model size in group init!"
        self.throughput = len(self.clients) * self.clients[0].model_size / \
                self.delay

    def __eq__(self, other):
        return self.aggregate_time == other.aggregate_time

    def __ne__(self, other):
        return self.aggregate_time != other.aggregate_time

    def __lt__(self, other):
        return self.aggregate_time < other.aggregate_time

    def __le__(self, other):
        return self.aggregate_time <= other.aggregate_time

    def __gt__(self, other):
        return self.aggregate_time > other.aggregate_time

    def __ge__(self, other):
        return self.aggregate_time >= other.aggregate_time


class AsyncServer(Server):
    """Asynchronous federated learning server."""

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.async_save_model(self.model, model_path, 0.0)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_clients(self, num_clients):
        super().make_clients(num_clients)

        # Set link speed for clients
        speed = []
        for client in self.clients:
            client.set_link(self.config)
            speed.append(client.speed_mean)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

        # Initiate client profile of loss and delay
        self.profile = Profile(num_clients)

    # Run asynchronous federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        # Init async parameters
        self.alpha = self.config.sync.alpha
        self.staleness_func = self.config.sync.staleness_func

        # Init self accuracy records
        self.records = Record()

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        T_old = 0.0
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Perform async rounds of federated learning with certain
            # grouping strategy
            self.rm_old_models(self.config.paths.model, T_old)
            accuracy, T_new = self.async_round(round, T_old)

            # Update time
            T_old = T_new

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def async_round(self, round, T_old):
        """Run one async round for T_async"""
        import fl_model  # pylint: disable=import-error
        target_accuracy = self.config.fl.target_accuracy

        # Select clients to participate in the round
        sample_groups = self.selection()
        sample_clients = []
        for group in sample_groups:
            for client in group.clients:
                client.set_delay()
                sample_clients.append(client)
            group.set_download_time(T_old)
            group.set_aggregate_time()
        self.throughput = sum([g.throughput for g in sample_groups])

        # Put the group into a queue according to its delay in ascending order
        # Each selected client will complete one local update in this async round
        queue = PriorityQueue()
        # This async round will end after the slowest group completes one round
        last_aggregate_time = max([g.aggregate_time for g in sample_groups])

        logging.info('Round lasts {} secs, avg throughput {} kB/s'.format(
            last_aggregate_time, self.throughput
        ))

        for group in sample_groups:
            queue.put(group)

        # Start the asynchronous updates
        while not queue.empty():
            select_group = queue.get()
            select_clients = select_group.clients
            self.async_configuration(select_clients, select_group.download_time)

            threads = [Thread(target=client.run(reg=True)) for client in select_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
            T_cur = select_group.aggregate_time  # Update current time
            logging.info('Training finished on clients {} at time {} s'.format(
                select_clients, T_cur
            ))

            # Receive client updates
            reports = self.reporting(select_clients)

            # Update profile and plot
            self.update_profile(reports)
            self.profile.plot('pf_{}.png'.format(T_cur))

            # Perform weight aggregation
            logging.info('Aggregating updates from clients {}'.format(select_clients))
            staleness = select_group.aggregate_time - select_group.download_time
            updated_weights = self.aggregation(reports, staleness)

            # Load updated weights
            fl_model.load_weights(self.model, updated_weights)

            # Extract flattened weights (if applicable)
            if self.config.paths.reports:
                self.save_reports(round, reports)

            # Save updated global model
            self.async_save_model(self.model, self.config.paths.model, T_cur)

            # Test global model accuracy
            if self.config.clients.do_test:  # Get average accuracy from client reports
                accuracy = self.accuracy_averaging(reports)
            else:  # Test updated model on server
                testset = self.loader.get_testset()
                batch_size = self.config.fl.batch_size
                testloader = fl_model.get_testloader(testset, batch_size)
                accuracy = fl_model.test(self.model, testloader)

            logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
            self.records.append_record(T_cur, accuracy, self.throughput)
            # Return when target accuracy is met
            if target_accuracy and \
                    (self.records.get_latest_acc() >= target_accuracy):
                logging.info('Target accuracy reached.')
                return self.records.get_latest_acc(), self.records.get_latest_t()

            # Insert the next aggregation of the group into queue
            # if time permitted
            if T_cur + select_group.delay < last_aggregate_time:
                select_group.set_download_time(T_cur)
                select_group.set_aggregate_time()
                queue.put(select_group)

        return self.records.get_latest_acc(), self.records.get_latest_t()


    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round
        select_type = self.config.clients.selection

        if select_type == 'random':
            # Select clients randomly
            sample_clients = [client for client in random.sample(
                self.clients, clients_per_round)]

        elif select_type == 'short_latency_first':
            # Select the clients with short latencies and random loss
            sample_clients = sorted(self.clients, key=lambda c:c.est_delay)
            sample_clients = sample_clients[:clients_per_round]
            print(sample_clients)

        elif select_type == 'short_latency_high_loss_first':
            # Get the non-negative losses and delays
            losses = [c.loss for c in self.clients]
            losses_norm = [l/max(losses) for l in losses]
            delays = [c.est_delay for c in self.clients]
            delays_norm = [d/max(losses) for d in delays]

            # Sort the clients by jointly consider latency and loss
            gamma = 0.2
            sorted_idx = sorted(range(len(self.clients)),
                                key=lambda i: losses_norm[i] - gamma * delays_norm[i],
                                reverse=True)
            print([losses[i] for i in sorted_idx])
            print([delays[i] for i in sorted_idx])
            sample_clients = [self.clients[i] for i in sorted_idx]
            sample_clients = sample_clients[:clients_per_round]
            print(sample_clients)

        # Create one group for each selected client to perform async updates
        sample_groups = [Group([client]) for client in sample_clients]

        return sample_groups

    def async_configuration(self, sample_clients, download_time):
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

            # Continue configuration on client
            client.async_configure(config, download_time)

    def aggregation(self, reports, staleness=None):
        return self.federated_async(reports, staleness)

    def extract_client_weights(self, reports):
        # Extract weights from reports
        weights = [report.weights for report in reports]

        return weights

    def federated_async(self, reports, staleness):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        weights = self.extract_client_weights(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        new_weights = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in weights[0]]
        for i, update in enumerate(weights):
            num_samples = reports[i].num_samples
            for j, (_, weight) in enumerate(update):
                # Use weighted average by number of samples
                new_weights[j] += weight * (num_samples / total_samples)

        # Extract baseline model weights - latest model
        baseline_weights = fl_model.extract_weights(self.model)

        # Calculate the staleness-aware weights
        alpha_t = self.alpha * self.staleness(staleness)
        logging.info('{} staleness: {} alpha_t: {}'.format(
            self.staleness_func, staleness, alpha_t
        ))

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append(
                (name, (1 - alpha_t) * weight + alpha_t * new_weights[i])
            )

        return updated_weights

    def staleness(self, staleness):
        if self.staleness_func == "constant":
            return 1
        elif self.staleness_func == "polynomial":
            a = 0.5
            return pow(staleness+1, -a)
        elif self.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)

    def async_save_model(self, model, path, download_time):
        path += '/global_' + '{}'.format(download_time)
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def rm_old_models(self, path, cur_time):
        for filename in os.listdir(path):
            try:
                model_time = float(filename.split('_')[1])
                if model_time < cur_time:
                    os.remove(os.path.join(path, filename))
                    logging.info('Remove model {}'.format(filename))
            except Exception as e:
                logging.debug(e)
                continue

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay)