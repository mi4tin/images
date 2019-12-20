from skimage.measure import compare_ssim
import cv2

imageA = cv2.imread("E:\Work\DEV\Code\SRC\github.com\images\data\original.png")
imageB = cv2.imread('E:\Work\DEV\Code\SRC\github.com\images\data\small.jpg')

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
print("SSIM: {}".format(score))