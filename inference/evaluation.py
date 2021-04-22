from time import time
import numpy as np
import paddle
from utils import *
from inference.transforms import Clicker
from tqdm import tqdm
import scipy.misc as sm
import os

def evaluate_dataset(dataset, predictor, oracle_eval=False, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        item = dataset[index]
        sample = dataset.get_sample(index)

        if oracle_eval:
            gt_mask = paddle.to_tensor(sample['instances_mask'], dtype='float32')
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            predictor.opt_functor.mask_loss.set_gt_mask(gt_mask)
        #print('item[images]', item[0].shape)
        #print('sample[instances_mask]', sample['instances_mask'].shape)
        _, sample_ious, _ = evaluate_sample(item[0], sample['instances_mask'], predictor, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


# def evaluate_sample(image_nd, instances_mask, predictor, max_iou_thr,
#                     pred_thr=0.49, max_clicks=20):
#     clicker = Clicker(gt_mask=instances_mask)
#     pred_mask = np.zeros_like(instances_mask)
#     ious_list = []

#     with paddle.no_grad():
#         predictor.set_input_image(image_nd)
#         for click_number in range(max_clicks):
#             clicker.make_next_click(pred_mask)
#             pred_probs = predictor.get_prediction(clicker)
#             pred_mask = pred_probs > pred_thr
#             iou = get_iou(instances_mask, pred_mask)
#             ious_list.append(iou)

#             if iou >= max_iou_thr:
#                 break

#         return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def evaluate_sample(image_nd, instances_mask, predictor, max_iou_thr,
                    pred_thr=0.49, max_clicks=20):
    clicker = Clicker(gt_mask=instances_mask)
    pred_mask = np.zeros_like(instances_mask)
    ious_list = []

    predictor.set_input_image(image_nd)
    pred_probs = None
    for click_number in range(max_clicks):
        clicker.make_next_click(pred_mask)
        pred_probs = predictor.get_prediction(clicker, pred_probs)
        pred_mask = pred_probs > pred_thr
        
        iou = get_iou(instances_mask, pred_mask)
        ious_list.append(iou)

        if iou >= max_iou_thr:
            image_name = str(time()) + '.png'
            sm.imsave(os.path.join('result', image_name), pred_probs) 
            break

    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
