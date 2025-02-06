
def diceLoss(prediction_g, label_g, epsilon=1):
    diceLabel_g = label_g.sum(dim=[1, 2, 3])
    dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])
    diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])

    diceRatio_g = (2 * diceCorrect_g + epsilon) \
                  / (dicePrediction_g + diceLabel_g + epsilon)

    return 1 - diceRatio_g