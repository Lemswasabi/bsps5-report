import librosa
import numpy as np
import pandas as pd
import os
import csv

numbers = '0 1 2 3 4 5 6 7 8 9'.split()

header = ''
for i in range(1, 641):
    header += f'mfcc{i} '
header += ' label'
header = header.split()

file = open('largetrainingset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for number in numbers:
    for filename in os.listdir(f'./recordings/{number}'):
        nfile = f'./recordings/{number}/{filename}'

        y, sr = librosa.load(nfile, sr=None, mono=True, duration=1)

        if y.size < 16000:
            rest = 16000 - y.size
            left = rest // 2
            right = rest - left

            y = np.pad(y, (left, right), 'reflect')

        mfccs = librosa.feature.mfcc(y=y, sr=sr)

        row = mfccs.T.flatten()
        row = np.append(row,number)

        file = open('largetrainingset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(row)
