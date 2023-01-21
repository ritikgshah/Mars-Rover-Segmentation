import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2hsv, rgb2luv, hsv2rgb, luv2rgb, rgb2lab, lab2rgb, rgb2gray
from functools import reduce
from PIL import Image
from scipy import linalg
import matlab.engine
from skimage import exposure, data, io, segmentation, color
from skimage.segmentation import felzenszwalb, mark_boundaries, slic
from plot_rag_merge import _weight_mean_color, merge_mean_color
from skimage.future import graph


def k_means(img, n_clusters):
    og_shape = img.shape
    img_data = img.reshape(-1, 3)
    img_data = np.float32(img_data)
    clf = KMeans(n_clusters=n_clusters).fit(img_data)
    labels = clf.labels_
    img_mod = np.reshape(labels, og_shape[:2])
    img_mod = np.uint8(img_mod)
    return img_mod

def mean_shift(img, quantile, n_samples):
    og_shape = img.shape
    img_data = img.reshape(-1, 3)
    img_data = np.float32(img_data)
    bandwidth = estimate_bandwidth(img_data, quantile=quantile, n_samples=n_samples) 
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True).fit(img_data)
    labels = ms.labels_
    img_mod = np.reshape(labels, og_shape[:2])
    img_mod = np.uint8(img_mod)
    return img_mod


# def decorrstretch(A):
#     eng = matlab.engine.start_matlab()
#     img3 = eng.imread('1059ML0046560000306154E01_DRCL.tif')
#     # decorr = eng.decorrstretch(img3, 'tol', tol)
#     decorr = eng.decorrstretch(img3)
#     return np.asarray(decorr)


def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
  
def linear_stretch(img):
    r1 = 5
    s1 = 0
    r2 = 250
    s2 = 255

    pixelVal_vec = np.vectorize(pixelVal)
    return pixelVal_vec(img, r1, s1, r2, s2)


def histogram_equalization(img):
    return exposure.equalize_hist(img, nbins=256)


def main():
    
    # Read the image
    # img = plt.imread('0073MR0003970000103657E01_DRCL.tif')
    # img = plt.imread('0174ML0009370000105185E01_DRCL.tif')
    img = plt.imread('0617ML0026350000301836E01_DRCL.tif')
    # img = plt.imread('1059ML0046560000306154E01_DRCL.tif')
    # img = decorrstretch(img)
    og_shape = img.shape
    img_data = img.reshape(-1, 3)
    # img = rgb2hsv(img)
    # img = linear_stretch(img)
    # img[...,0] = np.max(img[...,0]) - img[...,0]
    # Display the image
    # img = np.uint8(img)
    # img=histogram_equalization(img)
    plt.imshow(img)
    plt.show()
    
    # img_seg = felzenszwalb(img, scale=400, sigma=0.25, min_size=250)
    img_seg = slic(img, compactness=30, n_segments=4000)
    # img_seg = img_seg.reshape(img.shape)
    # img_seg = rgb2gray(img_seg)
    plt.imshow(mark_boundaries(img, img_seg))
    plt.title('Oversegmented felzenszwalb')
    plt.show()
    out = color.label2rgb(img_seg, img, kind='avg', bg_label=0)
    # out = segmentation.mark_boundaries(out, img_seg, (0, 0, 0))
    # plt.imshow(out)
    # plt.show()
    # g = graph.rag_mean_color(img, img_seg)
    # labels2 = graph.merge_hierarchical(img_seg, g, thresh=50, rag_copy=False,
    #                                    in_place_merge=True, merge_func=merge_mean_color,
    #                                    weight_func=_weight_mean_color)
    # graph.show_rag(img_seg, g, img)
    # plt.title('RAG after hierarchical merging')

    # plt.figure()
    # out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    # # out = segmentation.mark_boundaries(out, img_seg, (0, 0, 0))
    # plt.imshow(mark_boundaries(img, labels2))
    # plt.title('Oversegmented felzenszwalb with RAG and hierarchical merge')
    # plt.show()
    # plt.imshow(out)
    # plt.show()
    img_seg2 = k_means(out, 3)
    # img_seg2 = mean_shift(out, 0.35, 200)
    plt.imshow(mark_boundaries(img, img_seg2))
    plt.title('Mean shift clustering on top of oversegmented SLIC')
    plt.show()
    plt.imshow(img_seg2)
    plt.show()
    
    # img_seg3 = mean_shift(out, 0.3, 500)
    # # img_seg=histogram_equalization(img_seg)
    
    # plt.imshow(mark_boundaries(img, img_seg3))
    # plt.title('Mean shift clustering on top of oversegmented felzenszwalb')
    # plt.show()
    
    # plt.imshow(img_seg2)
    # plt.show()
    
    # img1 = cv.imread('0073MR0003970000103657E01_DRCL.tif')
    # img1_seg = img1[:,95:1500,:] 
    # mask = np.zeros(img1_seg.shape[:2],np.uint8)
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    # rect = (1,1,1200,900)
    # cv.grabCut(img1_seg,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # img1_seg = img1_seg*mask2[:,:,np.newaxis]
    # plt.imshow(img1_seg)
    # plt.title('GrabCut segmentation')
    # plt.show()

    img2 = cv.imread('0174ML0009370000105185E01_DRCL.tif')
    img2_seg = img2 
    mask = np.zeros(img2_seg.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,500,576)
    cv.grabCut(img2_seg,mask,rect,bgdModel,fgdModel,6,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img2_seg = img2_seg*mask2[:,:,np.newaxis]
    plt.imshow(img2_seg)
    plt.title('GrabCut segmentation')
    plt.show()


    # img3 = cv.imread('0617ML0026350000301836E01_DRCL.tif')
    # img3_seg = img3[:,95:1500,:]
    # mask = np.zeros(img3_seg.shape[:2],np.uint8)
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    # rect = (500,300,1400,1200)
    # cv.grabCut(img3_seg,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # img3_seg = img3_seg*mask2[:,:,np.newaxis]
    # plt.imshow(img3_seg)
    # plt.title('GrabCut segmentation')
    # plt.show()

    # img4 = cv.imread('1059ML0046560000306154E01_DRCL.tif')
    # img4_seg = img4
    # mask = np.zeros(img4_seg.shape[:2],np.uint8)
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    # rect = (0,0,800,1354)
    # cv.grabCut(img4_seg,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # img4_seg = img4_seg*mask2[:,:,np.newaxis]
    # plt.imshow(img4_seg)
    # plt.title('GrabCut segmentation')
    # plt.show()

if __name__ == '__main__':
    main()