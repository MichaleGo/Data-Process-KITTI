'''
目的：处理语义分割所需要的数据，产生RGB2DDS（两张RGB，disp，由dips计算得到的depth， senmantic）九个通道的图像

原始数据：
    From:
        RGB channel：Kitti data_scene_flow
        disp channnel：Kitti data_scene_flow
        semantic channel: Kitti
    To：RGB2DDS

转换公式： depth = baseline * focal / disparity
For KITTI the baseline is 0.54m and the focal ~721 pixels.

Note: total 200 pictures
'''

from PIL import Image
import numpy as np
import os


def gen_rgb2dds(rgb1_img_path,
                rgb2_img_path,
                disp_img_path,
                semantic_img_path,
                focal=721.0):
    # disp
    disp_png = np.array(Image.open(disp_img_path), dtype=np.uint16)
    assert (np.max(disp_png) > 255)
    disp = disp_png.astype(np.float16) / 256.0
    disp[disp_png == 0] = -1

    # rgb
    rgb1 = np.array(Image.open(rgb1_img_path), dtype=np.uint8)
    rgb2 = np.array(Image.open(rgb2_img_path), dtype=np.uint8)

    tup1 = rgb1.shape[0]
    tup2 = rgb1.shape[1]

    if tup1 > 375:
        tup1 = 375
    if tup2 > 1242:
        tup2 = 1242

    # depth
    depth = 0.54 * focal / disp
    depth.astype(np.float16)
    depth[depth < 0] = 0

    # semantic
    semantic = np.array(Image.open(semantic_img_path), dtype=np.int)

    # 判断深度图与rgb图像尺寸是否相同
    assert (rgb1.shape[0] == disp.shape[0] and rgb1.shape[1] == disp.shape[1]
            and rgb1.shape[0] == semantic.shape[0]
            and rgb1.shape[0] == semantic.shape[0])

    rgb2dds = np.zeros((375, 1242, 9), dtype=np.float16)
    rgb2dds[0:tup1, 0:tup2, 0:3] = rgb1[0:tup1, 0:tup2, :]
    rgb2dds[0:tup1, 0:tup2, 3:6] = rgb2[0:tup1, 0:tup2, :]
    rgb2dds[0:tup1, 0:tup2, 6] = disp[0:tup1, 0:tup2]
    rgb2dds[0:tup1, 0:tup2, 7] = depth[0:tup1, 0:tup2]
    rgb2dds[0:tup1, 0:tup2, 8] = semantic[0:tup1, 0:tup2]

    return rgb2dds


def get_focal(i):
    # get the mapping from sceneflow devkit
    dict = {
        '2011_09_26': 721.5377,
        '2011_09_28': 707.0493,
        '2011_09_29': 718.3351,
        '2011_09_30': 707.0912,
        '2011_10_03': 718.8560
    }

    if i >= 0 and i <= 154:
        return dict['2011_09_26']
    elif i == 155 or i == 156:
        return dict['2011_09_28']
    elif i >= 157 and i <= 171:
        return dict['2011_09_29']
    elif i >= 172 and i <= 198:
        return 721.0  # 具体情况不详 ？？？？？？
    else:
        return dict['2011_10_03']


def main():
    semantic_dir_path = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/single_view_depth/data_semantics/training/semantic'
    rgb1_dir_path = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/single_view_depth/data_scene_flow/training/image_2'
    rgb2_dir_path = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/single_view_depth/data_scene_flow/training/image_3'
    disp_dir_path = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/single_view_depth/data_scene_flow/training/disp_noc_0'
    dest_dir_pre = '/media/dongyanmei/2cb88bc6-316c-44b1-b0fd-4fc8fbf2c17f/xiangjun/npy'

    semantic_img_files = os.listdir(semantic_dir_path)
    semantic_img_files.sort(key=lambda x: int(x[:-4]))
    for i, name in enumerate(semantic_img_files):
        print(i, name)
        rgb1_img_path = os.path.join(rgb1_dir_path, name)
        rgb2_img_path = os.path.join(rgb2_dir_path, name)
        disp_img_path = os.path.join(disp_dir_path, name)
        semantic_img_path = os.path.join(semantic_dir_path, name)

        focal = get_focal(i)

        np.save(
            os.path.join(dest_dir_pre, 'semantic', name[:-4] + '.npy'),
            gen_rgb2dds(rgb1_img_path, rgb2_img_path, disp_img_path,
                        semantic_img_path, focal))


if __name__ == '__main__':
    main()
