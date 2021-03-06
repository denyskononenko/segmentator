#%%
# this script is for visualization of lunet-kernel.py comparator function action

import numpy as np
import cv2
import matplotlib.pyplot as plt 
from skimage.io import imread
import re

%run lunet-kernel.py

test_img1 = "https://img.lun.ua/construction-1600x900/21006.jpg"
test_img2 = "https://img.lun.ua/construction-1600x900/67081.jpg"

def extract_testimgs():
    """
    Extract urls of image pairs to compare from test_images.dat
    """
    str_mod = lambda a: re.search('"(.+?)"', a).group(1)
    f = open("/Users/denyskononenko/Documents/build_proc/test_images.dat", "r")
    data = f.readlines()
    data = list(filter(lambda x: x != '\n', data))
    new_data = [[str_mod(data[i]), str_mod(data[i+1])] for i in np.arange(0, len(data), 2)]
    return new_data

def find_similarity(image, image_base)->float:
    """
    Function for comparing two images with SIFT method
    
    @param image: path for image for comparison with base image
    @param image_base: path for base image to compare with
    @return similarity: similarity of corresponding images real number in range [0; 100]
    """
    
    img = imread(image)
    img_base = imread(image_base)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    
    # compare shape and resize if needed
    print("1: {}, 2: {}".format(gray_base.shape[0:2], gray_base.shape[0:2][::-1]))
    if gray.shape[0:2] != gray_base.shape[0:2]:
        gray = cv2.resize(gray, dsize=gray_base.shape[0:2][::-1], interpolation=cv2.INTER_CUBIC)
    
    sift = cv2.xfeatures2d.SIFT_create();
    kp_base, desc_base = sift.detectAndCompute(gray_base, None)
    kp, desc = sift.detectAndCompute(gray, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch( desc, desc_base,  k=2 )
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.95*n.distance:
            good.append([m])
    
    features_12 = [len(kp_base), len(kp)]
    similarity = 100 * len(good) / len(matches)
    
    print("similarity: {}".format(similarity))
    print("matches {}".format(len(matches)))
    print("length kp1 {}, lenght descr1 {}".format(len(kp), len(desc)))
    print("lenght kp {}, lenght descr {}".format(len(kp_base), len(desc_base)))
    return float('{0:.2f}'.format(similarity))


def calc_show_img_masks(im1, im2, new_sim=0, old_sim=0, image_size = 128):
    """
    Show images and their generated masks
    @param im1, im2 urls of images to display
    @param new_sim similarity between two images obtained by new u-net based method, 
                   if not mentioned equlas zero
    @param old_sim similarity between two images obtained by old pure sift based method,
                    if not mentioned equlas zero
    @param iamge_size size of images default 128
    """
    # extract masks for images
    image1 = cv2.resize(imread(im1), (image_size, image_size))
    image2 = cv2.resize(imread(im2), (image_size, image_size))
    msk1 = mask_ejector(image1)
    msk2 = mask_ejector(image2)
   
    # plot images and masks in grid 
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.subplot2grid((2, 4), (0, 0), colspan=1)
    plt.title("Image 1", fontsize=12)
    plt.imshow(image1)

    plt.subplot2grid((2, 4), (0, 1), colspan=1)
    plt.title("Mask 1", fontsize=12)
    plt.imshow(np.reshape(msk1, (image_size, image_size)), cmap="gray")

    plt.subplot2grid((2, 4), (0, 2), colspan=1)
    plt.title("Image 2", fontsize=12)
    plt.imshow(image2)

    plt.subplot2grid((2, 4), (0, 3), colspan=1)
    plt.title("Mask 2", fontsize=12)
    plt.imshow(np.reshape(msk2, (image_size, image_size)), cmap="gray")

    plt.subplot2grid((2, 4), (1, 0), colspan=2)
    plt.title("Overlapping 1", fontsize=12)
    plt.imshow(np.reshape(msk1, (image_size, image_size)), cmap="gray")
    plt.imshow(image1, cmap="jet", alpha=0.5)

    plt.subplot2grid((2, 4), (1, 2), colspan=2)
    plt.title("Overlapping 2", fontsize=12)
    plt.imshow(np.reshape(msk2, (image_size, image_size)), cmap="gray")
    plt.imshow(image2, cmap="jet", alpha=0.5)

    # add textbox with results of comparators
    plt.text(0, 150, "New comparator: {}".format(new_sim), size=20, rotation=0.,
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=("#16120a"),
                   fc=("#e4d9c4"),
                   alpha=0.5,
                   )
         )

    plt.text(0, 170, "Old comparator: {}".format(old_sim), size=20, rotation=0.,
         ha="left", va="center",
         bbox=dict(boxstyle="round",
                   ec=("#16120a"),
                   fc=("#e4d9c4"),
                   alpha=0.5,
                   )
         )


#%%
# test of new algorithm
new_sim = comparator(test_img1, test_img2)
#calc_show_img_masks(test_img2, test_img1)
# test of old algorithm
#old_sim = find_similarity(test_img1 ,test_img2)

#%%
print("Old version of comparator: {}".format(old_sim))
print("New version of comparator: {}".format(new_sim))
calc_show_img_masks(test_img2, test_img1, new_sim, old_sim)

#%%
# test on the selected examples
test_imgs_arr = extract_testimgs()

test_img1 = test_imgs_arr[0][0]
test_img2 = test_imgs_arr[0][1]
new_sim = comparator(test_img1, test_img2)
old_sim = find_similarity(test_img1 ,test_img2)
calc_show_img_masks(test_img2, test_img1, new_sim, old_sim)


#%%
image1 = cv2.resize(imread(test_img1), (128, 128))
image2 = cv2.resize(imread(test_img2), (128, 128))
msk1 =  mask_ejector(image1)
msk2 =  mask_ejector(image2)

resgr1 = np.uint8(np.reshape(msk1, (128, 128)))
resgr2 = np.uint8(np.reshape(msk2, (128, 128)))
    
canny1 = 255 - cv2.Canny(resgr1, 0, 0)
canny2 = 255 - cv2.Canny(resgr2, 0, 0)

plt.subplot(121)
plt.imshow(canny1, cmap="gray")
plt.subplot(122)
plt.imshow(canny2, cmap="gray")

#%%
sift = cv2.xfeatures2d.SIFT_create()
kp_base, desc_base = sift.detectAndCompute(canny1, None)
kp, desc = sift.detectAndCompute(canny2, None)
print(len(kp_base))

#%%
#matcher(test_img1, test_img2)
corr = signal.correlate2d(resgr1, resgr2, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(corr), corr.shape)
#print(x)
#print(y)
plt.imshow(corr)


#%%
print(np.max(corr, axis=0))