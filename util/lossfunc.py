
def diceLoss(prediction_g, label_g, epsilon=1):
    # dice = 2TP/(FN+FP+2TP) = 2真阳性/(实际阳性(TP+FN)+预测阳性(TP+FP))
    # 这里使用的是预测的值logits计算，没有转换成真值，因为做了比较后无法计算梯度

    # 将预测值压缩成(B,W,H),不要通道维度
    prediction_g = prediction_g.squeeze(dim=1)

    diceLabel_g = label_g.sum(dim=[-2, -1])
    dicePrediction_g = prediction_g.sum(dim=[-2, -1])
    diceCorrect_g = (prediction_g * label_g).sum(dim=[-2, -1])

    # epsilon是为了防止除数为0的情况
    diceRatio_g = (2 * diceCorrect_g + epsilon) \
                  / (dicePrediction_g + diceLabel_g + epsilon)

    return 1 - diceRatio_g