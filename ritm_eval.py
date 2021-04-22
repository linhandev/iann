import os
import pickle
import argparse
from pathlib import Path
from model import get_hrnet_model, get_deeplab_model, get_shufflenet_model
from paddleseg.utils import logger
from utils import *
from inference.predictor import get_predictor
from inference.evaluation import evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
#     parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,COCO_MVal,SBD',
#                         help='List of datasets on which the model should be tested. '
#                              'Datasets are separated by a comma. Possible choices: '
#                              'GrabCut, Berkeley, DAVIS, COCO_MVal, SBD')
    parser.add_argument('--datasets', type=str, default='Berkeley',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, COCO_MVal, SBD')
    parser.add_argument('--n-clicks', type=int, default=15,
                        help='Maximum number of clicks for the NoC metric.')

    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--target-iou', type=float, default=0.90,
                        help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    parser.add_argument('--norm-radius', type=int, default=260)
    parser.add_argument('--clicks-limit', type=int, default=10)
    parser.add_argument('--logs-path', type=str, default='log_output',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')


    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    #model = get_hrnet_model(width=18, ocr_width=64, with_aux_output=True, small=True, is_ritm=True)
    #model = get_deeplab_model(backbone='resnet18', is_ritm=True)
    model = get_shufflenet_model(is_ritm=True)
    #model = get_deeplab_model(backbone='mobilenet', is_ritm=True)
    #para_state_dict = paddle.load(args.checkpoint)
#     with open('checkpoint1.txt', 'w') as f:
#         for keys, values in model.state_dict().items():
#             f.write(keys +'\t'+str(values.shape)+"\n")
            
#     with open('checkpoint2.txt', 'w') as f:
#         for keys, values in para_state_dict.items():
#             f.write(keys +'\t'+str(values.shape)+"\n")
    
    #model.set_dict(para_state_dict)
    logger.info('Loaded trained params of model successfully')
    model.eval()

    eval_exp_name = get_eval_exp_name(args)
    eval_exp_path = os.path.join(args.logs_path, eval_exp_name)
    if not os.path.exists(eval_exp_path):
        os.makedirs(eval_exp_path)
    
    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params = {'net_clicks_limit': args.clicks_limit}

    print_header = True
    for dataset_name in args.datasets.split(','):
        dataset = get_dataset(dataset_name, args)
        zoom_in_target_size = 600 if dataset_name == 'DAVIS' else 400
        predictor = get_predictor(model, args.mode,
                                  prob_thresh=args.thresh,
                                  zoom_in_params={'target_size': zoom_in_target_size},
                                  predictor_params=predictor_params)
        dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                           max_iou_thr=args.target_iou,
                                           max_clicks=args.n_clicks)
        save_results(args, dataset_name, eval_exp_path, dataset_results,
                     print_header=print_header)
        print_header = False


def save_results(args, dataset_name, eval_exp_path, dataset_results, print_header=True):
    all_ious, elapsed_time = dataset_results
    mean_spc, mean_spi = get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)
    header, table_row = get_results_table(noc_list, over_max_list, args.mode, dataset_name,
                                          mean_spc, elapsed_time, args.n_clicks,
                                          model_name=eval_exp_path)
    target_iou_int = int(args.target_iou * 100)
    if target_iou_int not in [80, 85, 90]:
        noc_list, over_max_list = compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                     max_clicks=args.n_clicks)
        table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
        table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)
    log_path = f'results_{args.mode}_{args.n_clicks}.txt'
    log_path = os.path.join(eval_exp_path, log_path)
    

    #log_path = eval_exp_path / f'results_{args.mode}_{args.n_clicks}.txt'
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            f.write(header + '\n')
            f.write(table_row + '\n')

    ious_path = os.path.join(eval_exp_path, 'all_ious')
    if not os.path.exists(ious_path):
        os.makedirs(ious_path)
    with open(os.path.join(ious_path, f'{dataset_name}_{args.mode}_{args.n_clicks}.pkl'), 'wb') as fp:
        pickle.dump(all_ious, fp)

if __name__ == '__main__':
    main()