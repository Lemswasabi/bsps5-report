from tqdm import tqdm
from pydub import AudioSegment
from joblib import Parallel, delayed
from python_speech_features import logfbank, fbank, mfcc

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav


def init_parser():
    parser = argparse.ArgumentParser(description='Librispeech preprocess.')

    parser.add_argument('root', metavar='root', type=str,
                        help='Absolute file path to LibriSpeech. (e.g. /usr/downloads/LibriSpeech/)')

    parser.add_argument('train_set', metavar='train_set', type=str,
                        help='Training datasets to process in LibriSpeech. (e.g. train-clean-100/)')

    parser.add_argument('--dev_set', metavar='dev_set', type=str,
                        help='Validation datasets to process in LibriSpeech. (e.g. dev-clean/)')

    parser.add_argument('--test_set', metavar='test_set', type=str,
                        help='Testing datasets to process in LibriSpeech. (e.g. test-clean/)')

    parser.add_argument('--n_jobs', dest='n_jobs', action='store', default=-2 ,
                        help='number of cpu availible for preprocessing.\n -1: use all cpu, -2: use all cpu but one')

    parser.add_argument('--n_filters', dest='n_filters', action='store', default=40,
                        help='Number of filters for fbank. (Default : 40)')

    parser.add_argument('--win_size', dest='win_size', action='store', default=0.025,
                        help='Window size during feature extraction (Default : 0.025 [25ms])')

    parser.add_argument('--norm_x', dest='norm_x', action='store', default=False,
                        help='Normalize features s.t. mean = 0 std = 1')

    parser.add_argument('--speech_feature', dest='speech_feature', action='store', default='fbank',
                        help='Speech feature for feature extration (Default : fbank)')
    return parser

def parse_labels(root, dataset_dir):
    labels = []
    dataset_path = os.path.join(root, dataset_dir)
    for speaker in sorted(os.listdir(dataset_path)):
        for chapter in sorted(os.listdir(os.path.join(dataset_path, speaker))):
            text_filepath = os.path.join(dataset_path, speaker, chapter)
            text_filename = '{speaker}-{chapter}.trans.txt'.format(speaker=speaker, chapter=chapter)
            with open(os.path.join(text_filepath, text_filename), 'r') as f:
                lines = f.read().splitlines()
                labels.extend([' '.join(line.split(' ')[1:]).lower() for line in lines])
    return labels

def parse_audio(root, dataset_dir, search_filetype='.flac'):
    audio_files = []
    dataset_path = os.path.join(root, dataset_dir)
    for speaker in sorted(os.listdir(dataset_path)):
        for chapter in sorted(os.listdir(os.path.join(dataset_path, speaker))):
            for audio_file in sorted(os.listdir(os.path.join(dataset_path, speaker, chapter))):
                if audio_file.endswith(search_filetype):
                    audio_files.append(os.path.join(dataset_path, speaker, chapter, audio_file))
    return audio_files

def flac2wav(f_path):
    flac_audio = AudioSegment.from_file(f_path, "flac")
    flac_audio.export(f_path[:-4] + 'wav', format="wav")

def wav2logfbank(f_path):
    (rate, sig) = wav.read(f_path)
    fbank_feat = logfbank(sig, rate, winlen=win_size, nfilt=n_filters)
    filename = f_path[:-4] + '-logfbank' + str(n_filters)
    np.save(filename, fbank_feat)
    return filename + '.npy'

def wav2mfcc(f_path):
    (rate, sig) = wav.read(f_path)
    fbank_feat = mfcc(sig, rate, winlen=win_size, nfilt=n_filters)
    filename = f_path[:-4] + '-mfcc' + str(n_filters)
    np.save(filename, fbank_feat)
    return filename + '.npy'

def norm(f_path, mean, std):
    np.save(f_path, (np.load(f_path)-mean)/std)

def process_flac2wav():
    global train_file_list, dev_file_list, test_file_list

    print('Processing flac2wav')

    print('Training')
    train_file_list = parse_audio(root, train_path)
    _ = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(f) for f in tqdm(train_file_list))

    print('Validation')
    dev_file_list = parse_audio(root, dev_path)
    _ = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(f) for f in tqdm(dev_file_list))

    print('Testing')
    test_file_list = parse_audio(root, test_path)
    _ = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(f) for f in tqdm(test_file_list))

def process_wav2speech_feature(speech_feature):
    global train_file_list, dev_file_list, test_file_list

    speech_features = {
        'logfbank': wav2logfbank,
        'mfcc': wav2mfcc,
    }

    wav2feature = speech_features.get(speech_feature)
    print('Processing wav2{speech_feature}'.format(speech_feature=speech_feature))
    print('Training')
    train_file_list = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2feature)(f[:-4] + 'wav') for f in tqdm(train_file_list))
    print('Validation')
    dev_file_list = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2feature)(f[:-4] + 'wav') for f in tqdm(dev_file_list))
    print('Testing')
    test_file_list = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2feature)(f[:-4] + 'wav') for f in tqdm(test_file_list))

def process_speech_feature2dataset(dataset_file_list, dataset_path, output_filename):
    global mean_x, std_x
    # dataset_file_list = parse_audio(root, dataset_path, search_filetype='.npy')
    dataset_text = parse_labels(root, dataset_path)

    X = []
    for f in dataset_file_list:
        X.append(np.load(f))

    # Normalize X
    if norm_x:
        if mean_x == None and std_x == None:
            mean_x = np.mean(np.concatenate(X, axis=0), axis=0)
            std_x = np.std(np.concatenate(X, axis=0), axis=0)
        _ = Parallel(n_jobs=n_jobs,backend="threading")(delayed(norm)(f, mean_x, std_x) for f in tqdm(data_file_list))

    # Sort data by signal length (long to short)
    audio_len = [len(x) for x in X]
    dataset_file_list = [dataset_file_list[idx] for idx in reversed(np.argsort(audio_len))]
    dataset_text = [dataset_text[idx] for idx in reversed(np.argsort(audio_len))]

    # text to index sequence
    tmp_list = []
    for text in dataset_text:
        build_vocab(text)
        tmp = [str(VOCAB[char]) for char in text]
        tmp_list.append(tmp)
    dataset_text = tmp_list
    del tmp_list

    output_filepath = os.path.join(root, output_filename)

    print('Writing dataset to {output_filepath}'.format(output_filepath=output_filepath))

    dataset_text = [' '.join(sentence) for sentence in dataset_text]

    dataset = {
        'input': dataset_file_list,
        'label': dataset_text
    }

    df = pd.DataFrame(dataset)
    df.to_csv(output_filepath, index=False)

def build_vocab(text):
    global VOCAB, IVOCAB
    for token in text:
        for char in token:
            if char not in VOCAB:
                next_index = len(VOCAB)
                VOCAB[char] = next_index
                IVOCAB[next_index] = char

def save_vocab():
    global IVOCAB
    IVOCAB = pd.DataFrame(list(IVOCAB.items()), columns=['key', 'value'])
    IVOCAB.to_csv(os.path.join(root, 'VOCAB.csv'), index=False)

if __name__ == '__main__':

    paras = init_parser().parse_args()

    root = paras.root
    train_path = paras.train_set
    dev_path = paras.dev_set
    test_path = paras.test_set
    n_jobs = paras.n_jobs
    n_filters = paras.n_filters
    win_size = paras.win_size
    norm_x = paras.norm_x
    speech_feature = paras.speech_feature

    VOCAB = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    IVOCAB = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
    mean_x = None
    std_x = None

    print('----------Processing Datasets----------')

    print('Training sets :', train_path)
    print('Validation sets :', dev_path)
    print('Testing sets :', test_path)

    print('---------------------------------------')

    process_flac2wav()

    print('---------------------------------------')

    process_wav2speech_feature('logfbank')

    print('---------------------------------------')

    print('Preparing Training Dataset')

    process_speech_feature2dataset(train_file_list, train_path, 'train.csv')

    print('Preparing Validation Dataset')

    process_speech_feature2dataset(dev_file_list, dev_path, 'dev.csv')

    print('Preparing Testing Dataset')

    process_speech_feature2dataset(test_file_list, test_path, 'test.csv')

    print('Saving VOCAB')

    save_vocab()
