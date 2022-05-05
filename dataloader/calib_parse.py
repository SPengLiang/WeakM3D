import numpy as np


def parse_calib(mode, calib_path=None):
    if mode == '3d':
        with open(calib_path, encoding='utf-8') as f:
            text = f.readlines()
            P0 = np.array(text[0].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P1 = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(text[3].split(' ')[1:], dtype=np.float32).reshape(3, 4)

            Tr_velo_to_cam = np.array(text[5].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            # Tr_imu_to_velo = np.array(text[6].split(' ')[1:], dtype=np.float32).reshape(3, 4)

            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[4].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1

            Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
            '''lidar to image pixel plane'''
            l2p = np.dot(np.dot(P2, R_rect), Tr_velo_to_cam)
            '''lidar to image plane'''
            l2i = np.dot(R_rect_tmp, Tr_velo_to_cam[:3])

    elif mode == 'raw':
        calib_cam2cam_path, velo2cam_calib_path = calib_path
        with open(velo2cam_calib_path, encoding='utf-8') as f:
            text = f.readlines()
            R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

            trans = np.concatenate([R, T], axis=1)
            vel2cam = trans.copy()

            Tr_velo_to_cam = np.concatenate([trans, np.array([[0, 0, 0, 1]])], axis=0)

        with open(calib_cam2cam_path, encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[-9].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[8].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1

            '''lidar to image pixel plane'''
            l2p = np.dot(np.dot(P2, R_rect), Tr_velo_to_cam)
            '''lidar to image plane'''
            l2i = np.dot(R_rect_tmp, vel2cam)

    calib = {
        'P2': P2,
        'l2p': l2p,
        'l2i': l2i
    }
    return calib