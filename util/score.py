import numpy as np


def diceScore(gt, pred):
    return 2 * np.sum(gt * pred) / (np.sum(gt) + np.sum(pred))

def iouScore(gt, pred):
    return np.sum(gt * pred) / np.sum(gt + pred - gt * pred)