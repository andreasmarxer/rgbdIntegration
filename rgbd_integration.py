# examples/Python/Advanced/rgbd_integration.py

import open3d as o3d
from trajectory_io import *
import numpy as np
import time

if __name__ == "__main__":
    # =========================== settings ====================================
    bag_name = 'original5'

    #pose_path = '/data/' + bag_name +'/30-Nov-2019-11h-51rgbdPose_open3D.log' # original5 NOT optimized
    pose_path = '/data/' + bag_name +'/06-Dec-2019-21h-20rgbdPose_open3D_OPTIMIZED.log'  # original5 optimized

    #asl_train_labels = [1,16,39,90,128,178,199,221,264,269,283,289,307,337,350,355,361,363,365,368]
    asl_train_labels = [0]

    rgb_path = '/data/' + bag_name + '/rgb/'
    depth_path_original = '/data/' + bag_name + '/depth/'
    depth_path_median_kernel = '/data/' + bag_name + '/depth_median/'
    depth_path_adj = '/data/' +bag_name + '/depth_adj/plane_th50/'
    use_depth = 'adj'  # 'original' OR 'adj' OR 'median_kernel'
    visualization_range = [2,496] #4: 516, 5: 497, 6: 201, 7:193
    # ========================end of setting ==================================

    if use_depth == 'original':
        print("- Use original depth")
    elif use_depth == 'median_kernel':
        print("- Use median kernel depth")
    elif use_depth == 'adj':
        print("- Use adjusted depth")

    pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
        "data/camera_intrinsic_asl.json")

    camera_poses = read_trajectory(pose_path)
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.03, #0.03 Marius  #4.0 / 512.0,  # 512 -> = 0.007812
        sdf_trunc=0.12,    #0.12 Marius  #0.04
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    for i in range(visualization_range[0], visualization_range[1]):
        if i in asl_train_labels:
            print("{:d}-th Image used for train - skipped.".format(i))
            continue # skip this iteration

        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(
            rgb_path + 'asl_window_rgb_{:d}.jpg'.format(i))
        if use_depth == 'original':
            depth = o3d.io.read_image(
                depth_path_original + 'asl_window_depth_{:d}.png'.format(i))
        elif use_depth == 'median_kernel':
            depth = o3d.io.read_image(
                depth_path_median_kernel + 'asl_window_depth_median5_{:d}.png'.format(i))
        elif use_depth == 'adj':
            depth = o3d.io.read_image(
                depth_path_adj + 'asl_window_{:d}_depth_adj.png'.format(i))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            pinhole_camera_intrinsic,
            np.linalg.inv(camera_poses[i].pose))

    print("- Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    print("- Finished")

    # save mesh
    filename_out = 'mesh_' + use_depth + '_' + str(visualization_range[0]) + '-' + str(visualization_range[1]) +'_' + time.strftime("%Y%m%d-%H%M%S") + '.ply'
    o3d.io.write_triangle_mesh(filename_out, mesh)
