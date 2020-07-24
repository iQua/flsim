import argparse
from datetime import datetime
import re


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, help='Simmulation log file.')
args = parser.parse_args()

# Read log
with open(args.log, 'r') as f:
    log = [x for x in f.readlines() if x != '\n']

# Extract time
def extract_time(line):
    return datetime.strptime([x for x in re.split('\[|\]', line) if x][1], '%H:%M:%S')

# Extract lines
training = []

for line in log:
    if 'Round 1/' in line:
        training.append(line)

training.append(log[-1])

# Calculate duration
training_duration = (extract_time(training[1]) - extract_time(training[0])).seconds
print('{}: training time: {} s'.format(args.log, training_duration))
