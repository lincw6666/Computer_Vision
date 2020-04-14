import numpy as np
import cv2
import sys
import argparse
from convolution import convolution


img_name = {
    'jpg': ['cathedral', 'monastery', 'nativity', 'tobolsk'],
    'tif': ['emir', 'icon', 'lady', 'melons', 'onion_church',
            'three_generations', 'train', 'village', 'workshop']
}


# Image loader.
class ImageLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __iter__(self):
        for img_type in img_name.keys():
            for fname in img_name[img_type]:
                yield self.cut_image(
                    cv2.imread(self.root_dir + fname + '.' + img_type))

    def __getitem__(self, idx):
        img_type = idx[0]
        fname = img_name[img_type][idx[1]]
        return self.cut_image(
            cv2.imread(self.root_dir + fname + '.' + img_type))

    def cut_image(self, origin_img):
        # Since it's a grayscale image, we can obtain the intensity by
        # extracting one of its channel.
        img = origin_img[:, :, 0]
        
        # The cut appears roughly at 1/3 of image height. We don't need to
        # check for the whole image.
        anchor = [img.shape[0] // 3 * i for i in range(1, 3)]
        offset = [img.shape[0] // 50, img.shape[1] // 4]
        col_mid = img.shape[1] // 2
        
        crop_img = [img[_anchor - offset[0]: _anchor + offset[0],
                        col_mid - offset[1]: col_mid + offset[1], None]\
                    for _anchor in anchor]
        
        # Find the cut.
        cut = [0]
        tmp = [np.argmin(crop_img[i].sum(axis=1)) + anchor[i] - offset[0]\
            for i in range(len(crop_img))]
        cut.extend(tmp)
        cut.append(origin_img.shape[0])
        
        return [origin_img[cut[i]:cut[i+1], :, :] for i in range(len(cut)-1)]


# Sum of square distance.
def SSD(img1, img2):
    return np.sqrt(((img1-img2)**2).sum())


def visualization(img):
    window_name = f'Result {img.shape}'
    shape = (img.shape[1], img.shape[0])
    scale = 1024 / max(shape) if max(shape) > 1024 else 1.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(shape[0]*scale), int(shape[1]*scale))
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Get arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='hw2_data/task3_colorizing/',
                        help='path to hw2 task03 data')
    args = parser.parse_args()
    if args.data_dir is None:
        print('Error!! Missing arguments!!')
        parser.print_help()
        sys.exit(0)
    elif args.data_dir[-1] != '/':
        args.data_dir += '/'

    # Read image.
    img_loader = ImageLoader(args.data_dir)
    for img in [img_loader['tif', i] for i in range(len(img_name['tif']))]:
        for _img in img:
            visualization(_img)