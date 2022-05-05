import os
import torch
import argparse
import sys
sys.path.append(os.getcwd())

from dataloader import build_dataloader
from lib import network
from config import cfg
from scripts.train import eval_one_epoch


def evaluation(cfg):
    layer = cfg.NET_LAYER
    restore_path = cfg.RESTORE_PATH
    dim_prior = cfg.DATA.DIM_PRIOR
    gt_dir = cfg.VAL.GT_DIR

    save_dir_exp = os.path.join(cfg.INFER.SAVE_DIR,
                                os.path.splitext(os.path.basename(restore_path))[0] + '/data')
    print('Predictions saved in : {}'.format(save_dir_exp))

    model = network.ResnetEncoder(num_layers=layer)
    model.load_state_dict(torch.load(restore_path))
    model.cuda()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    InferImgLoader_RoI = build_dataloader.build_infer_loader(cfg)

    eval_one_epoch(save_dir_exp, InferImgLoader_RoI, model, dim_prior, gt_dir, ap_mode=40)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    evaluation(cfg)