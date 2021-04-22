import paddle
import os
import numpy as np
import pickle


def train():
    model1 = paddle.load('pretrained/MobileNetV3_large_x1_0_pretrained.pdparams')
#     with open('MobileNetV3_large_x1_0_pretrained.txt', 'w') as f:
#         for keys, values in model1.items():
#             f.write(keys +'\t'+str(values.shape)+"\n")

    predict = {}

    for keys, values in model1.items():
        model_key = 'feature_extractor.backbone.' + keys
        x = model1[keys]
        predict[model_key] = x
        print(model_key)

    params_output = open('MobileNetV3_large_x1_0_pretrained.pdparams', 'wb')
    pickle.dump(predict, params_output)

if __name__ == '__main__':
    train()
