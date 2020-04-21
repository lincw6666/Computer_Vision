import numpy as np
import cv2
import sys
import argparse
import matplotlib.pyplot as plt
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
                yield fname, self.cut_image(
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
        
        return [origin_img[cut[i]:cut[i+1], :, 0] for i in range(len(cut)-1)]


# Sum of square distance.
def SSD(img1, img2):
    if img1.shape != img2.shape:
        return -1
    return ((img1-img2)**2).sum()


def visualization(img):
    assert isinstance(img, np.ndarray),\
        "[visualization] Image should be a ndarray!!"
    assert 2 <= len(img.shape) <= 3,\
        f"[visualization] Image has wrong dimension: {len(img.shape)}."
    if len(img.shape) == 2:
        img = np.stack([img for _ in range(3)], axis=2)

    window_name = f'Result {img.shape}'
    shape = (img.shape[1], img.shape[0])
    
    # Scale to fit our monitor.
    scale = 1024 / max(shape) if max(shape) > 1024 else 1.

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(shape[0]*scale), int(shape[1]*scale))
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_merge_image(img, offset, save_filename=None):
    # Define the origin of the merged images.
    rgb_origin = offset.min(axis=0)
    # Found the height and width of the merged images.
    rgb_h, rgb_w = 0, 0
    for i, _offset in enumerate(offset):
        rgb_h = max(rgb_h, img[i].shape[0] + _offset[0] - rgb_origin[0])
        rgb_w = max(rgb_w, img[i].shape[1] + _offset[1] - rgb_origin[1])
    # Initialize the merged images.
    rgb_img = np.zeros((rgb_h, rgb_w, 3))

    # Fill each channel in the image.
    for i, _offset in enumerate(offset):
        start_h = _offset[0] - rgb_origin[0]
        start_w = _offset[1] - rgb_origin[1]
        rgb_img[start_h:start_h+img[i].shape[0],
                start_w:start_w+img[i].shape[1], i] = img[i]
    rgb_img = rgb_img.astype(np.uint8)
    if save_filename is not None:
        cv2.imwrite(save_filename, rgb_img)
    plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    plt.show()


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
    # for img in [img_loader['jpg', i] for i in range(len(img_name['jpg']))]:
    for fname, img in img_loader:
        # @img: a list contains 3 images, which belong to 3 different channels.
        #       These 3 images have shape (h, w), without channel dimension.
        gaussian_pyramid = []
        for i, _img in enumerate(img):
        
            # Generate Gaussian pyramid.
            G = img[i].copy()
            tmp = [G]
            for i in range(int(np.log2(max(_img.shape))) - 6):
                G = cv2.pyrDown(G)
                tmp.append(G)
            
            # Usage: @gaussian_pyramid[channel][level]
            gaussian_pyramid.append(tmp)

        # Align the images.

        # Offset of channel 1->3, 2->3, 3->3
        offset = np.array([[0, 0], [0, 0], [0, 0]])
        offset_range = 15

        # Only compute the SSD of the center 80% of the image.
        h = int(min([gaussian_pyramid[i][-1].shape[0] for i in range(3)]) * .8)
        w = int(min([gaussian_pyramid[i][-1].shape[1] for i in range(3)]) * .8)

        # Top left corner of the center of img2.
        start_h = (gaussian_pyramid[-1][-1].shape[0]-h) // 2
        start_w = (gaussian_pyramid[-1][-1].shape[1]-w) // 2
        
        kernel = np.array([[ 0, -1,  0],
                           [-1,  4, -1],
                           [ 0, -1,  0]])

        for level in range(len(gaussian_pyramid[0])-1, -1, -1):
            # We compare B and G to R channel. Therefore, we fix R channel as
            # @img2.
            img2 = np.squeeze(
                convolution(gaussian_pyramid[-1][level][:, :, None],
                            kernel)
            )

            # Lower 1 level makes the image 2 times bigger. Therefore, we need
            # to scale @h, @w, and @offset 2 times bigger.
            if level != len(gaussian_pyramid[0])-1:
                offset = offset * 2
                h, w = h * 2, w * 2

            for channel in range(2):
                img1 = np.squeeze(
                    convolution(gaussian_pyramid[channel][level][:, :, None],
                                kernel)
                )
                
                # Iterate the offset range to align the images.
                min_ssd = 255 * h * w
                new_offset = offset[channel]
                now_offset_range = min(offset_range, min(h//8, w//8))
                for i in range(-now_offset_range, now_offset_range+1):
                    for j in range(-now_offset_range, now_offset_range+1):
                        tmp_h = start_h - (offset[channel][0] + i)
                        tmp_w = start_w - (offset[channel][1] + j)
                        tmp = SSD(img1[tmp_h:tmp_h+h, tmp_w:tmp_w+w],
                                  img2[start_h:start_h+h, start_w:start_w+w])
                        if 0 <= tmp < min_ssd:
                            min_ssd = tmp
                            new_offset = offset[channel] + np.array([i, j])
                offset[channel] = new_offset
            offset_range = 1
        
        # Plot the result.
        visualize_merge_image(img, offset, save_filename=f'{fname}.tif')
