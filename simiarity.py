from skimage.measure import compare_ssim
import cv2

#imageA = cv2.imread("E:\Work\DEV\Code\SRC\github.com\images\data\mm.jpg")
#imageB = cv2.imread('E:\Work\DEV\Code\SRC\github.com\images\data\mf.jpg')
imageA = cv2.imread("E:\Work\DEV\Code\SRC\github.com\images\data\icon1.png")
imageB = cv2.imread('E:\Work\DEV\Code\SRC\github.com\images\data\icon1.png')

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
print("SSIM: {}".format(score))