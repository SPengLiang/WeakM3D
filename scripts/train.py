import cv2 as cv
import numpy as np
import torch.nn.functional as F
import torch
import sys
from tqdm import tqdm
import torch.optim as optim
import os
import loguru
import argparse

sys.path.append(os.getcwd())

from utils import log
from dataloader import build_dataloader
from lib import network
from lib import loss_factory
from config import cfg
from utils import eval
from utils import post_improve_3d


def train(cfg):
    exp_name = cfg.EXP_NAME
    layer = cfg.NET_LAYER
    restore_epoch = cfg.RESTORE_EPOCH
    lr = cfg.TRAIN.LR
    epochs = cfg.TRAIN.EPOCH
    dim_prior = cfg.DATA.DIM_PRIOR
    gt_dir = cfg.VAL.GT_DIR

    log.prepare_dirs(cfg)
    log.init_logger(cfg)
    logger = loguru.logger

    model = network.ResnetEncoder(num_layers=layer)
    model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    start_epoch = 0
    if restore_epoch:
        restore_path_pkl = os.path.join(cfg.CHECKPOINTS_DIR, exp_name+'_'+str(restore_epoch)+'.pkl')
        model.load_state_dict(torch.load(restore_path_pkl), strict=False)
        opt_restore_path_pkl = os.path.join(cfg.CHECKPOINTS_DIR, exp_name+'_optim_'+str(restore_epoch)+'.pkl')
        optimizer.load_state_dict(torch.load(opt_restore_path_pkl))
        start_epoch = restore_epoch

    viz_dict = {
        'vis_cls': 0,
        'vis_count': 0,
        'epoch_cls': 0,
    }

    global_step = 0
    TrainImgLoader_RoI = build_dataloader.build_train_loader(cfg)
    InferImgLoader_RoI = build_dataloader.build_infer_loader(cfg)

    logger.info('Start training')

    for epoch_idx in range(start_epoch, epochs):
        model.train()
        for batch_idx, sample in enumerate(TrainImgLoader_RoI):
            global_step = 1 + batch_idx + len(TrainImgLoader_RoI) * epoch_idx

            batch_input = build_dataloader.process_batch_data(sample)

            pred_3D = model(batch_input['l_img'], batch_input['bbox2d'])

            loss = loss_factory.build_loss(pred_3D,
                                         batch_input['batch_RoI_points'],
                                         batch_input['batch_lidar_y_center'],
                                         batch_input['batch_lidar_orient'],
                                         batch_input['batch_lidar_density'],
                                         batch_input['P2'],
                                         batch_input['bbox2d'],
                                         batch_input['batch_dim']
                                         )
            if loss is None:
                print('no valid loss at: ', global_step, pred_3D[0][:, 1])
                continue

            optimizer.zero_grad()
            loss.backward()
            nan_flag = 0

            for name, parms in model.named_parameters():
                if name in ['module.location_z.4.weight', 'module.location_z.4.bias',
                            'location_z.4.weight', 'location_z.4.bias']:
                    if torch.sum(parms.grad != parms.grad) > 0:
                        logger.warning('loss back NAN, ignore! continue training')
                        nan_flag = 1

            if not nan_flag:
                optimizer.step()


            viz_dict['vis_cls'] += float(loss)
            viz_dict['epoch_cls'] += float(loss)
            viz_dict['vis_count'] += 1

            if viz_dict['vis_count'] % 100 == 0 and viz_dict['vis_count'] > 0:
                logger.info(
                    "Epoch_idx: {}, global_step: {}, loss: {:.4f}, max: {} epochs".format(
                        epoch_idx, global_step, float(viz_dict['vis_cls'] / 100), cfg.TRAIN.EPOCH
                    )
                )
                viz_dict['vis_cls'] = 0

        logger.info("Epoch: {}; Average loss: {}".format(epoch_idx,
                                                      viz_dict['epoch_cls'] /len(TrainImgLoader_RoI)))
        viz_dict['epoch_cls'] = 0

        checkpoints_path = os.path.join(cfg.CHECKPOINTS_DIR, '{}_{}.pkl'.format(exp_name, epoch_idx))
        optim_path = os.path.join(cfg.CHECKPOINTS_DIR, '{}_optim_{}.pkl'.format(exp_name, epoch_idx))
        logger.info(
            "Saving checkpoint at {}. Epoch: {}, Global_step: {}".format(
                checkpoints_path, epoch_idx, global_step
            )
        )
        torch.save(model.state_dict(), checkpoints_path)
        torch.save(optimizer.state_dict(), optim_path)


        ###########################################################################
        # Evaluation
        ###########################################################################
        save_dir_exp = os.path.join(cfg.INFER.SAVE_DIR,
                                    os.path.splitext(os.path.basename(checkpoints_path))[0] + '/data')
        eval_one_epoch(save_dir_exp, InferImgLoader_RoI, model, dim_prior, gt_dir, ap_mode=40)



def eval_one_epoch(save_dir_exp, InferImgLoader_RoI, model, dim_prior, gt_dir, ap_mode=40):
    if not os.path.exists(save_dir_exp):
        os.makedirs(save_dir_exp)
    with torch.no_grad():
        model.eval()
        for batch_idx, sample in tqdm(enumerate(InferImgLoader_RoI)):
            batch_input = build_dataloader.process_batch_data(sample)
            P2 = batch_input['P2'][0].cpu().numpy()
            bbox2d = batch_input['bbox2d'][0].cpu().numpy()
            det_2D = batch_input['det_2D'][0].cpu().numpy()
            file_name = batch_input['file_name'][0]

            if bbox2d.shape[0] < 1:
                np.savetxt('{}/{}.txt'.format(save_dir_exp, file_name), np.array([]), fmt='%s')
                continue

            pred_3D = model(batch_input['l_img'], batch_input['bbox2d'])

            p_locxy, p_locZ, p_ortConf = pred_3D
            p_locXYZ = torch.cat([p_locxy, p_locZ], dim=1)

            fx, fy, cx, cy = P2[0][0], P2[1][1], P2[0][2], P2[1][2]

            det_3D = np.zeros((p_locXYZ.shape[0], 16), dtype=object)
            det_3D[:, 0] = ['Car' for _ in range(p_locXYZ.shape[0])]
            det_3D[:, 4:8] = det_2D[:, 1:5]
            det_3D[:, -1] = det_2D[:, -1]
            '''car dimension'''
            det_3D[:, 8:11] = [np.array(dim_prior[2]) for _ in range(p_locXYZ.shape[0])]

            for i in range(len(p_locXYZ)):
                p, b = p_locXYZ[i], det_2D[i, 1:5]
                h, w, center_x, center_y = b[3] - b[1], b[2] - b[0], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                proj_box_center = ((F.sigmoid(p[:2]) - 0.5) * torch.tensor([w, h]).cuda() + \
                                   torch.tensor([center_x, center_y]).cuda() - \
                                   torch.tensor([cx, cy]).cuda()) / torch.tensor([fx, fy]).cuda()
                proj_box_center = torch.cat([proj_box_center, torch.tensor([1.]).cuda()])
                location_3d = p[2] * proj_box_center
                det_3D[i, 11:14] = location_3d.cpu().numpy()

                alpha_ratio = F.normalize((p_ortConf[i].unsqueeze(0))).squeeze(0)
                estimated_theta = torch.atan2(alpha_ratio[0], alpha_ratio[1])
                det_3D[i, 3] = float(estimated_theta)

                det_3D[i, 12] += float(det_3D[i, 8]) / 2
                det_3D[i, -2] = det_3D[i, 3] + np.arctan2(det_3D[i, 11], det_3D[i, 13])

            det_3D[:, 1:] = np.around(det_3D[:, 1:].astype(np.float64), decimals=5)
            np.savetxt('{}/{}.txt'.format(save_dir_exp, file_name), det_3D, fmt='%s')
        post_improve_3d.post_3d(save_dir_exp, save_dir_exp)
        eval.eval_from_scrach(gt_dir, save_dir_exp, ap_mode=11)
        eval.eval_from_scrach(gt_dir, save_dir_exp, ap_mode=40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Training model")
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

    train(cfg)
