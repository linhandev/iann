import argparse
import tkinter as tk
from model import get_hrnet_model, get_deeplab_model
import paddle


from util import exp
import utils
from interactive_demo.app import InteractiveDemoApp


def main():
    args, cfg = parse_args()
    #model = get_hrnet_model(width=18, ocr_width=64, small=True, with_aux_output=True, is_ritm=True,cpu_dist_maps=True)
    model = get_deeplab_model(backbone='resnet18', is_ritm=True, cpu_dist_maps=True)

    para_state_dict = paddle.load('/home/aistudio/git/paddle/ritm/human_best/resnet18_ritm_95.5/model.pdparams')
    model.set_dict(para_state_dict)
    print('Loaded trained params of model successfully')
    model.eval()
    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--norm-radius', type=int, default=260)

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == '__main__':
    main()
