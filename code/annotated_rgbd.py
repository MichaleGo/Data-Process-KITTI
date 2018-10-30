'''
目的：处理单目视觉实验所需要的数据，产生RGBD（depth_data_annotated）四个通道的图像

原始数据：
    From:
        depth channel：Kitti： depth completion -> data_depth_annotated 数据集
        RGB channnel：KITTI raw data
    To：RGBD

Note: train 有142个文件夹， val 有13个文件夹, 总共155文件夹， 少于KITTI: raw data 总共156个文件夹
'''

from PIL import Image
import numpy as np
import os


def gen_rgbd(depth_img_path, rgb_img_path):
    # depth
    depth_png = np.array(Image.open(depth_img_path), dtype=np.uint16)         # uint16 等数据类型的问题
    assert (np.max(depth_png) > 255)
    depth = depth_png.astype(np.float16) / 256
    # depth[depth_png == 0] = -1

    # rgb
    rgb = np.array(Image.open(rgb_img_path), dtype=np.uint16)

    tup1 = rgb.shape[0]
    tup2 = rgb.shape[1]

    if tup1 > 375:
        tup1 = 375
    if tup2 > 1242:
        tup2 = 1242

    # 判断深度图与rgb图像尺寸是否相同
    assert (rgb.shape[0] == depth.shape[0]
            and rgb.shape[1] == depth.shape[1])

    rgbd = np.zeros((375, 1242, 4), dtype=np.float16)
    rgbd[0:tup1, 0:tup2, 0:3] = rgb[0:tup1, 0:tup2, :]
    rgbd[0:tup1, 0:tup2, 3] = depth[0:tup1, 0:tup2]

    return rgbd


def process_annotated():
    rgb_path_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/kitti_raw'
    depth_val_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/single_view_depth/data_depth_annotated/val'
    depth_train_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/single_view_depth/data_depth_annotated/train'
    dest_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/npy'

    depth_val_dir_list = os.listdir(depth_val_dir_pre)
    depth_train_dir_list = os.listdir(depth_train_dir_pre)

    for i, name in enumerate(depth_val_dir_list):
        print(i, depth_val_dir_list[i])
        assert (name == depth_val_dir_list[i])
        # 深度图某一个目录路经，目录下是照片
        depth_dir_path = os.path.join(depth_val_dir_pre, name,
                                      'proj_depth/groundtruth/image_02')
        # 与深度图目录对应的rgb目录
        rgb_dir_path = os.path.join(rgb_path_pre, name, name[0:10], name,
                                    'image_02/data')
        # 当前目录下所有照片的名字
        depth_img_files = os.listdir(depth_dir_path)
        for img in depth_img_files:
            depth_img_path = os.path.join(depth_dir_path, img)  # 当前深度照片的路经
            rgb_img_path = os.path.join(rgb_dir_path, img)  # 对应rgb照片的路径
            np.save(
                os.path.join(dest_dir_pre, 'annotated', 'val',
                             name + '_' + img[:-4] + '.npy'),
                gen_rgbd(depth_img_path, rgb_img_path))

    for i, name in enumerate(depth_train_dir_list):
        print(i, depth_train_dir_list[i])
        # 深度图某一个目录路经，目录下是照片
        depth_dir_path = os.path.join(depth_train_dir_pre, name,
                                      'proj_depth/groundtruth/image_02')
        # 与深度图目录对应的rgb目录
        rgb_dir_path = os.path.join(rgb_path_pre, name, name[0:10], name,
                                    'image_02/data')
        # 当前目录下所有照片的名字
        depth_img_files = os.listdir(depth_dir_path)
        for img in depth_img_files:
            depth_img_path = os.path.join(depth_dir_path, img)  # 当前深度照片的路经
            rgb_img_path = os.path.join(rgb_dir_path, img)  # 对应rgb照片的路径
            np.save(
                os.path.join(dest_dir_pre, 'annotated', 'train',
                             name + '_' + img[:-4] + '.npy'),
                gen_rgbd(depth_img_path, rgb_img_path))


def main():
    process_annotated()


if __name__ == '__main__':
    main()
