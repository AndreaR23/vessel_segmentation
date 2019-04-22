import numpy as np
from keras import backend as K


class Metrics:
    
    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (np.sum(np.square(y_true),-1) + np.sum(np.square(y_pred),-1) + smooth)

    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return 1 - Metrics.dice_coef(y_true, y_pred)

    @staticmethod
    def sensitivity(y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    @staticmethod
    def specificity(y_true, y_pred):
        true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = np.sum(np.round(np.clip(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())
