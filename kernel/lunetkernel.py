import keras 
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy import signal, ndimage

path_model = "/home/image_similarity/segmentator/saved_models/lunet-model-300-epochs.h5"
path_to_weights = "/home/image_similarity/segmentator/saved_models/UNet-params-300-epochs-new-data.h5"

def model_build(path_model, path_to_weights):
    """
    Build u-net model. Download model architecture and 
    hyperparameters values for it
    @param path_model path to the model architecture
    @param path_to_weights path to the model weights
    @return model keras model with u-net architecture with uplaoded hyperparameters
    """
    model = keras.models.load_model(path_model)
    model.load_weights(path_to_weights)
    return model

def mask_ejector(img, image_size = 128):
    """
    Eject mask of image with loaded u-net model
    @param img image array
    @return mask binary mask of building on the img
    """
    model = model_build(path_model, path_to_weights)
    mask = np.reshape(model.predict(np.array([img])), (image_size, image_size))
    # remove holes in the binary mask
    mask = ndimage.binary_fill_holes(mask).astype(int)
    return mask

def crop_img(img):
    """
    Crop image bottom 1/3 part to escape ground, cars, trees etc.
    @param img iamge array for croping
    @return cropped image
    """
    return img[:img.shape[0] - img.shape[0]//3, :]

def comparator(img1, img2):
    """
    Function for comparing two images with SIFT method applied for 
    binary masks of buildings on the images
    @param img1 compared image
    @param img2 image to comapre
    @return similarity real number in range [0; 100]
    """
    # crop bottom of images
    #image1 = crop_img(imread(img1))
    #image2 = crop_img(imread(img1))

    # resize and generate masks
    image1 = cv2.resize(imread(img1), (128, 128))
    image2 = cv2.resize(imread(img2), (128, 128))
    msk1 =  mask_ejector(image1)
    msk2 =  mask_ejector(image2)

    resgr1 = np.uint8(np.reshape(msk1, (128, 128)))
    resgr2 = np.uint8(np.reshape(msk2, (128, 128)))
    
    canny1 = 255 - cv2.Canny(resgr1, 0, 0)
    canny2 = 255 - cv2.Canny(resgr2, 0, 0)

    sift = cv2.xfeatures2d.SIFT_create()
    kp_base, desc_base = sift.detectAndCompute(canny1, None)
    kp, desc = sift.detectAndCompute(canny2, None)
    
    
    # BFMatcher with default params
    try:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch( desc, desc_base,  k=2 )
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.95*n.distance:
                good.append([m])

        print(len(matches))
        if len(matches) != 0:
            similarity = 100 * len(good) / len(matches)
        else:
            print("No matches")
            similarity = 0.0
    except Exception as e:
        print(e)
        similarity = 0.0
        matches = []
    
    print("similarity: {}".format(similarity))
    print("matches {}".format(len(matches)))
    #print("length kp1 {}, length descr1 {}".format(len(kp), len(desc)))
    #print("length kp {}, length descr {}".format(len(kp_base), len(desc_base)))
    return float('{0:.2f}'.format(similarity))

def matcher(img1, img2):
    """
    Find similarity between tow images with buildings by matching 
    their two binary masks of iamges via cross correlation.
    @param img1 compared image
    @param img2 image to comapre
    """
    # resize and generate masks
    image1 = cv2.resize(imread(img1), (128, 128))
    image2 = cv2.resize(imread(img2), (128, 128))
    msk1 =  mask_ejector(image1)
    msk2 =  mask_ejector(image2)

    resgr1 = np.uint8(np.reshape(msk1, (128, 128)))
    resgr2 = np.uint8(np.reshape(msk2, (128, 128)))

    # calculate cross correlation of two images  
    corr = signal.correlate2d(resgr1, resgr2, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    print(x)
    print(y)
    plt.plot(corr)


