### Takes in a Caffe log output from training, and extracts
### the training and testing loss at each iteration (multiple of 100).
### Dumps this to a .csv for local visualization.

import re
import pandas as pd


with open('parametric_init_log_20k-50k.txt', 'rb') as f:
    lines = f.read().split('\n')

train_losses = []
test_losses = []

for line in lines:
    if 'Test net output #0' in line:
        test_losses.append(float(
            line[line.index('loss = ') + 7: line.index('(') - 1]))
    elif 'Train net output #0' in line:
        train_losses.append(float(
            line[line.index('loss = ') + 7: line.index('(') - 1]))

df = pd.DataFrame({
    'iteration': xrange(201, 501),
    'training_loss': train_losses,
    'testing_loss': test_losses[:-1] # Extra last tesing iteration
    })

df.to_csv('parametric_init_losses_20k-50k.csv', index=False)