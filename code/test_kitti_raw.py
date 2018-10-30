from PIL import Image
import numpy as np


def main():

    depthfile = '/home/mcislab/gaoxiangjun/xiangjun/kitti_raw/data_depth_velodyne/val/2011_10_03_drive_0047_sync/proj_depth/velodyne_raw/image_02/0000000556.png'
    depth_png = np.array(Image.open(depthfile), dtype=int)
    depth = depth_png.astype(np.float32) / 256
    depth[depth_png == 0] = -1
    print('kitti_raw_depth')
    print(depth[200:205, 600:605])

    print('disp_png')
    dispfile = '/home/mcislab/gaoxiangjun/xiangjun/single_view_depth/data_scene_flow/training/disp_noc_0/000199_10.png'
    disp_png = np.array(Image.open(dispfile), dtype=np.uint16)
    disp = disp_png.astype(np.float32) / 256.0
    disp[disp_png == 0] = -1
    print('real_disp')
    print(disp[200:205, 600:605])

    disp_from_depth = 0.54 * 718.856 / (depth)
    disp_from_depth[depth < 0] = -1
    print('disp_from_depth')
    print(disp_from_depth[200:205, 600:605])


if __name__ == '__main__':
    main()
