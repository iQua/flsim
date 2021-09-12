import argparse
import client
import config
import logging
import os
import server
from datetime import datetime

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        # "dqn": server.DQNServer(fl_config), # DQN server disabled
        # "dqntrain": server.DQNTrainServer(fl_config), # DQN server disabled
        "sync": server.SyncServer(fl_config),
        "async": server.AsyncServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Save and plot accuracy-time curve
    if fl_config.server == "sync" or fl_config.server == "async":
        d_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        fl_server.record.save_acc_record('acc_{}_{}.csv'.format(
            fl_config.server, d_str
        ))
        fl_server.record.plot_acc_record('acc_{}_{}.png'.format(
            fl_config.server, d_str
        ))

    # Delete global model
    os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
