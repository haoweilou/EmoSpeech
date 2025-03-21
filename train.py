from data_utils import TextStyleDataset
from utils import get_hparams
hps = get_hparams()
dataset = TextStyleDataset("train.csv", hps)