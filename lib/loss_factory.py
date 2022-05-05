import torch
import numpy as np
import torch.nn.functional as F

from loguru import logger

def calc_dis_ray_tracing(wl, Ry, points, density, bev_box_center):
    init_theta, length = torch.atan(wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [((length * torch.cos(init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(np.pi - init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(np.pi - init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(np.pi + init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(np.pi + init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(-init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(-init_theta + Ry) + bev_box_center[1]).unsqueeze(0))]
    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4
    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4 # avoid inf
    #################################################
    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)


    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + torch.abs(intersect3[:, 1] - points[:, 1])


    dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)

    intersect0 = torch.round(intersect0*1e4)
    intersect1 = torch.round(intersect1*1e4)
    intersect2 = torch.round(intersect2*1e4)
    intersect3 = torch.round(intersect3*1e4)


    dis0_in_box_edge = ((intersect0[:, 0] > torch.round(min(corners[0][0], corners[1][0])*1e4)) &
                        (intersect0[:, 0] < torch.round(max(corners[0][0], corners[1][0])*1e4))) | \
                       ((intersect0[:, 1] > torch.round(min(corners[0][1], corners[1][1])*1e4)) &
                        (intersect0[:, 1] < torch.round(max(corners[0][1], corners[1][1])*1e4)))
    dis1_in_box_edge = ((intersect1[:, 0] > torch.round(min(corners[1][0], corners[2][0])*1e4)) &
                        (intersect1[:, 0] < torch.round(max(corners[1][0], corners[2][0])*1e4))) | \
                       ((intersect1[:, 1] > torch.round(min(corners[1][1], corners[2][1])*1e4)) &
                        (intersect1[:, 1] < torch.round(max(corners[1][1], corners[2][1])*1e4)))
    dis2_in_box_edge = ((intersect2[:, 0] > torch.round(min(corners[2][0], corners[3][0])*1e4)) &
                        (intersect2[:, 0] < torch.round(max(corners[2][0], corners[3][0])*1e4))) | \
                       ((intersect2[:, 1] > torch.round(min(corners[2][1], corners[3][1])*1e4)) &
                        (intersect2[:, 1] < torch.round(max(corners[2][1], corners[3][1])*1e4)))
    dis3_in_box_edge = ((intersect3[:, 0] > torch.round(min(corners[3][0], corners[0][0])*1e4)) &
                        (intersect3[:, 0] < torch.round(max(corners[3][0], corners[0][0])*1e4))) | \
                       ((intersect3[:, 1] > torch.round(min(corners[3][1], corners[0][1])*1e4)) &
                        (intersect3[:, 1] < torch.round(max(corners[3][1], corners[0][1])*1e4)))


    dis_in_mul = torch.stack([dis0_in_box_edge, dis1_in_box_edge,
                              dis2_in_box_edge, dis3_in_box_edge], dim=1)
    dis_inter2cen = torch.stack([dis_inter2center0, dis_inter2center1,
                                 dis_inter2center2, dis_inter2center3], dim=1)
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1)

    # dis_in = torch.sum(dis_in_mul, dim=1).type(torch.bool)
    dis_in = (torch.sum(dis_in_mul, dim=1) == 2).type(torch.bool)
    if torch.sum(dis_in.int()) < 3:
        return 0

    dis_in_mul = dis_in_mul[dis_in]
    dis_inter2cen = dis_inter2cen[dis_in]
    dis_all = dis_all[dis_in]
    density = density[dis_in]

    z_buffer_ind = torch.argmin(dis_inter2cen[dis_in_mul].view(-1, 2), dim=1)
    z_buffer_ind_gather = torch.stack([~z_buffer_ind.byte(), z_buffer_ind.byte()],
                                      dim=1).type(torch.bool)
    # z_buffer_ind_gather = torch.stack([~(z_buffer_ind.type(torch.bool)), z_buffer_ind.type(torch.bool)],
    #                                   dim=1)

    dis = (dis_all[dis_in_mul].view(-1, 2))[z_buffer_ind_gather] / density

    dis_mean = torch.mean(dis)
    return dis_mean



def calc_dis_rect_object_centric(wl, Ry, points, density):
    init_theta, length = torch.atan(wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [((length * torch.cos(init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(init_theta + Ry)).unsqueeze(0)),

               ((length * torch.cos(np.pi - init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(np.pi - init_theta + Ry)).unsqueeze(0)),

               ((length * torch.cos(np.pi + init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(np.pi + init_theta + Ry)).unsqueeze(0)),

               ((length * torch.cos(-init_theta + Ry)).unsqueeze(0),
                (length * torch.sin(-init_theta + Ry)).unsqueeze(0))]

    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4

    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    # line0 = [k1, -1, b11]
    # line1 = [k1, -1, b12]
    # line2 = [k2, -1, b21]
    # line3 = [k2, -1, b22]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4
    #################################################
    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)

    # dis0 = torch.sqrt((intersect0[:, 0] - points[:, 0])**2 +
    #                   (intersect0[:, 1] - points[:, 1])**2)
    # dis1 = torch.sqrt((intersect1[:, 0] - points[:, 0])**2 +
    #                   (intersect1[:, 1] - points[:, 1])**2)
    # dis2 = torch.sqrt((intersect2[:, 0] - points[:, 0])**2 +
    #                   (intersect2[:, 1] - points[:, 1])**2)
    # dis3 = torch.sqrt((intersect3[:, 0] - points[:, 0])**2 +
    #                   (intersect3[:, 1] - points[:, 1])**2)

    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + torch.abs(intersect3[:, 1] - points[:, 1])


    # dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    # dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    # dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    # dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)
    #
    # dis_point2center = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    #################################################

    points_z = torch.cat([points, torch.zeros_like(points[:, :1])], dim=1)
    vec0 = torch.tensor([corners[0][0], corners[0][1], 0])
    vec1 = torch.tensor([corners[1][0], corners[1][1], 0])
    vec2 = torch.tensor([corners[2][0], corners[2][1], 0])
    vec3 = torch.tensor([corners[3][0], corners[3][1], 0])

    ''' calc direction of vectors'''
    cross0 = torch.cross(points_z, vec0.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]
    cross1 = torch.cross(points_z, vec1.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]
    cross2 = torch.cross(points_z, vec2.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]
    cross3 = torch.cross(points_z, vec3.unsqueeze(0).repeat(points_z.shape[0], 1))[:, 2]


    ''' calc angle across vectors'''
    norm_p = torch.sqrt(points_z[:, 0] ** 2 + points_z[:, 1] ** 2)
    norm_d = torch.sqrt(corners[0][0] ** 2 + corners[0][1] ** 2).repeat(points_z.shape[0])
    norm = norm_p * norm_d

    dot_vec0 = torch.matmul(points_z, vec0.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec1 = torch.matmul(points_z, vec1.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec2 = torch.matmul(points_z, vec2.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]
    dot_vec3 = torch.matmul(points_z, vec3.unsqueeze(0).repeat(points_z.shape[0], 1).t())[:, 0]

    angle0 = torch.acos(dot_vec0/(norm))
    angle1 = torch.acos(dot_vec1/(norm))
    angle2 = torch.acos(dot_vec2/(norm))
    angle3 = torch.acos(dot_vec3/(norm))

    angle_sum0 = (angle0 + angle1) < np.pi
    angle_sum1 = (angle1 + angle2) < np.pi
    angle_sum2 = (angle2 + angle3) < np.pi
    angle_sum3 = (angle3 + angle0) < np.pi

    cross_dot0 = (cross0 * cross1) < 0
    cross_dot1 = (cross1 * cross2) < 0
    cross_dot2 = (cross2 * cross3) < 0
    cross_dot3 = (cross3 * cross0) < 0

    cross_all = torch.stack([cross_dot0, cross_dot1, cross_dot2, cross_dot3], dim=1)
    angle_sum_all = torch.stack([angle_sum0, angle_sum1, angle_sum2, angle_sum3], dim=1)

    choose_ind = cross_all & angle_sum_all
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1) / density.unsqueeze(1)
    choose_dis = dis_all[choose_ind]
    choose_dis[choose_dis != choose_dis] = 0

    return choose_dis



def build_loss(pred_3D, batch_RoI_points, batch_lidar_y_center,
                             batch_lidar_orient, batch_lidar_density, P2,
                             bbox2d, batch_dim):
    all_loss, count = 0, 0
    per_sample = batch_RoI_points.shape[1]
    fx = P2[:, 0, 0].view(-1, 1).repeat(1, per_sample).view(-1)
    fy = P2[:, 1, 1].view(-1, 1).repeat(1, per_sample).view(-1)
    cx = P2[:, 0, 2].view(-1, 1).repeat(1, per_sample).view(-1)
    cy = P2[:, 1, 2].view(-1, 1).repeat(1, per_sample).view(-1)

    p_locXY, p_locZ, p_ortConf = pred_3D


    bbox2d = bbox2d.view(-1, 4)
    p_locXY = p_locXY.view(-1, 2)
    p_locZ = p_locZ.view(-1, 1)
    p_ortConf = p_ortConf.view(-1, 2)
    p_locXYZ = torch.cat([p_locXY, p_locZ], dim=1)

    num_instance = bbox2d.shape[0]
    batch_RoI_points = batch_RoI_points.view(num_instance, -1, 3)
    batch_lidar_y_center = batch_lidar_y_center.view(num_instance)
    batch_lidar_orient = batch_lidar_orient.view(num_instance)
    batch_lidar_density = batch_lidar_density.view(num_instance, -1)
    abs_dim = batch_dim.view(num_instance, 3)

    h, w, center_x, center_y = bbox2d[:, 3] - bbox2d[:, 1], \
                               bbox2d[:, 2] - bbox2d[:, 0], \
                               (bbox2d[:, 0] + bbox2d[:, 2])/2, \
                               (bbox2d[:, 1] + bbox2d[:, 3])/2
    proj_box_center = ((F.sigmoid(p_locXYZ[:, :2])-0.5) * torch.stack([w, h], dim=1) + \
                 torch.stack([center_x, center_y], dim=1) - \
                 torch.stack([cx, cy], dim=1)) / torch.stack([fx, fy], dim=1)
    box_center = torch.cat([proj_box_center, torch.ones((proj_box_center.shape[0], 1)).cuda()], dim=1)
    location_3d = p_locXYZ[:, 2:3] * box_center

    alpha_ratio = F.normalize(p_ortConf, dim=1)
    Ry_pred = torch.atan2(alpha_ratio[:, 0], alpha_ratio[:, 1]) % np.pi

    Ry = batch_lidar_orient
    Alpha = -batch_lidar_orient
    trans_Ry = (torch.atan2(location_3d[:, 0], location_3d[:, 2])).detach()
    Alpha = (Alpha - trans_Ry) % np.pi


    for i in range(len(p_locXYZ)):
        single_Ry = Ry[i]
        single_wl = abs_dim[i, 1:]
        single_loc = location_3d[i]
        single_Alpha = Alpha[i]
        single_Ry_pred = Ry_pred[i]

        single_depth_points = batch_RoI_points[i]
        single_density = batch_lidar_density[i]
        single_lidar_center_y = batch_lidar_y_center[i]


        if single_loc[2] > 3:
            ray_tracing_loss = calc_dis_ray_tracing(single_wl, single_Ry, single_depth_points[:, [0, 2]], single_density,
                                                    (single_loc[0], single_loc[2]))
        else:
            ray_tracing_loss = 0


        shift_depth_points = torch.stack([single_depth_points[:, 0] - single_loc[0],
                                          single_depth_points[:, 2] - single_loc[2]], dim=1)
        dis_error = calc_dis_rect_object_centric(single_wl, single_Ry, shift_depth_points, single_density)
        dis_error = torch.mean(dis_error)

        # '''center_loss'''
        center_loss = torch.mean(torch.abs(shift_depth_points[:, 0]) / single_density) + \
                      torch.mean(torch.abs(shift_depth_points[:, 1]) / single_density)

        center_yloss = F.smooth_l1_loss(single_loc[1], single_lidar_center_y)

        # ''' LiDAR 3D box orient loss'''
        orient_loss = -1 * torch.cos(single_Alpha - single_Ry_pred)


        all_loss += dis_error + 0.1 * center_loss + ray_tracing_loss + \
                          center_yloss + orient_loss
        count += 1

    if count == 0:
        return None

    all_loss = all_loss / count
    return all_loss

