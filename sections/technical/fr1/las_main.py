import yaml
import torch
import argparse

parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
paras = parser.parse_args()
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

if paras.test:
    # Test ASR
    assert paras.load is None, 'Load option is mutually exclusive to --test'
    from bin.test_asr import Solver
    mode = 'test'
else:
    # Train ASR
    from bin.train_asr import Solver
    mode = 'train'

solver = Solver(config, paras, mode)
solver.load_data()
solver.set_model()
solver.exec()
