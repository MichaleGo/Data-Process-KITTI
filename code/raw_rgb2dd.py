'''
目的：处理双目视觉所需要的数据，产生RGB2DD（两张RGB，由depth计算得到的disp， depth）八个通道的图像

原始数据：
    From:
        RGB channel：Kitti raw_data kitti_raw
        depth channnel：Kitti depth_completion  data_depth_velodyne
    To：RGB2DD

转换公式： disparity = baseline * focal / depth
For KITTI the baseline is 0.54m and the focal ~721 pixels.

Note: data_depth_velodyne一共只有151（val 13 个， train 138 个）， 但是kitti_raw有156个目录， 缺少五个
'''

from PIL import Image
import numpy as np
import os


def gen_rgb2dd(rgb1_img_path, rgb2_img_path, depth_img_path, focal):
    # depth
    depth_png = np.array(
        Image.open(depth_img_path), dtype=np.uint16)  # uint16 等数据类型的问题
    assert (np.max(depth_png) > 255)
    depth = depth_png.astype(np.float16) / 256
    depth[depth_png == 0] = -1

    # rgb
    rgb1 = np.array(Image.open(rgb1_img_path), dtype=np.uint8)
    rgb2 = np.array(Image.open(rgb2_img_path), dtype=np.uint8)

    tup1 = rgb1.shape[0]
    tup2 = rgb1.shape[1]

    if tup1 > 375:
        tup1 = 375
    if tup2 > 1242:
        tup2 = 1242

    # 判断深度图与rgb图像尺寸是否相同
    assert (rgb1.shape[0] == depth.shape[0]
            and rgb1.shape[1] == depth.shape[1])

    # disp
    disp = 0.54 * focal / depth
    disp.astype(np.float16)
    disp[disp < 0] = 0

    rgb2dd = np.zeros((375, 1242, 8), dtype=np.float16)
    rgb2dd[0:tup1, 0:tup2, 0:3] = rgb1[0:tup1, 0:tup2, :]
    rgb2dd[0:tup1, 0:tup2, 3:6] = rgb2[0:tup1, 0:tup2, :]
    rgb2dd[0:tup1, 0:tup2, 6] = disp[0:tup1, 0:tup2]
    rgb2dd[0:tup1, 0:tup2, 7] = depth[0:tup1, 0:tup2]

    return rgb2dd


def get_focal(name):
    dict = {
        '2011_09_26': 721.5377,
        '2011_09_28': 707.0493,
        '2011_09_29': 718.3351,
        '2011_09_30': 707.0912,
        '2011_10_03': 718.8560
    }

    return dict[name[0:10]]


def main():
    depth_train_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/kitti_raw/data_depth_velodyne/train'
    depth_val_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/kitti_raw/data_depth_velodyne/val'
    rgb_path_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/kitti_raw'
    dest_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/npy'

    depth_val_dir_list = os.listdir(depth_val_dir_pre)
    depth_train_dir_list = os.listdir(depth_train_dir_pre)

    for i, name in enumerate(depth_val_dir_list):
        print(i, name)
        assert (name == depth_val_dir_list[i])

        # 深度图目录 路经，目录下是照片
        depth_dir_path = os.path.join(depth_val_dir_pre, name,
                                      'proj_depth/velodyne_raw/image_02')
        # 与深度图目录对应的 rgb目录
        rgb1_dir_path = os.path.join(rgb_path_pre, name, name[0:10], name,
                                     'image_02/data')
        rgb2_dir_path = os.path.join(rgb_path_pre, name, name[0:10], name,
                                     'image_03/data')

        # 当前目录下所有照片的名字
        depth_img_files = os.listdir(depth_dir_path)
        for img in depth_img_files:
            depth_img_path = os.path.join(depth_dir_path, img)  # 当前深度照片的路经
            rgb1_img_path = os.path.join(rgb1_dir_path, img)  # 对应rgb1照片的路径
            rgb2_img_path = os.path.join(rgb2_dir_path, img)  # 对应rgb2照片的路径

            focal = get_focal(name)

            np.save(
                os.path.join(dest_dir_pre, 'kitti_raw', 'val',
                             name + '_' + img[:-4] + '.npy'),
                gen_rgb2dd(rgb1_img_path, rgb2_img_path, depth_img_path,
                           focal))

    for i, name in enumerate(depth_train_dir_list):
        print(i, name)
        assert (name == depth_train_dir_list[i])

        # 深度图目录 路经，目录下是照片
        depth_dir_path = os.path.join(depth_train_dir_pre, name,
                                      'proj_depth/velodyne_raw/image_02')
        # 与深度图目录对应的 rgb目录
        rgb1_dir_path = os.path.join(rgb_path_pre, name, name[0:10], name,
                                     'image_02/data')
        rgb2_dir_path = os.path.join(rgb_path_pre, name, name[0:10], name,
                                     'image_03/data')

        # 当前目录下所有照片的名字
        depth_img_files = os.listdir(depth_dir_path)
        for img in depth_img_files:
            depth_img_path = os.path.join(depth_dir_path, img)  # 当前深度照片的路经
            rgb1_img_path = os.path.join(rgb1_dir_path, img)  # 对应rgb1照片的路径
            rgb2_img_path = os.path.join(rgb2_dir_path, img)  # 对应rgb2照片的路径

            focal = get_focal(name)

            np.save(
                os.path.join(dest_dir_pre, 'kitti_raw', 'train',
                             name + '_' + img[:-4] + '.npy'),
                gen_rgb2dd(rgb1_img_path, rgb2_img_path, depth_img_path,
                           focal))


if __name__ == '__main__':
    main()
