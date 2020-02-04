import logging
import os, re
import numpy as np
import matplotlib 
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from roipoly import MultiRoi
from roipoly import RoiPoly

path = './testset'
save_path = './test_masks/'

for filename in os.listdir(path):

    img = plt.imread(path + '/' + filename, format='jpg')

    # Show the image
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.title("Click on the button to add a new ROI")

    # Draw multiple ROIs
    multiroi_named = MultiRoi(roi_names=['My first ROI', 'My second ROI'])

    image_id = re.search("\d+", filename).group()

    # combine masks from multiple ROI
    mask = np.zeros((img.shape[0], img.shape[1]))
    for name, roi in multiroi_named.rois.items():
        mask += roi.get_mask(img[:,:,0])

    # save final mask
    plt.imsave(save_path + image_id + '.jpg', mask, cmap=cm.gray)


    
