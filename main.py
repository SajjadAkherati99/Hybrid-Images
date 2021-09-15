import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import cv2


def warp(im, H, out_shape):
    lent, wid = out_shape[0], out_shape[1]
    x = np.zeros((lent, wid, 3))
    H = np.linalg.inv(H)

    for i in range(0, lent):
        for j in range(0, wid):
            y = np.array([i, j, 1])
            transformed = np.dot(H, y.T)
            m, n = int(round(transformed[0])), int(round(transformed[1]))
            length_im, width_im = im.shape[0], im.shape[1]
            if 0 <= m < length_im and 0 <= n < width_im:
                x[i, j, :] = im[m, n, :]

    return x

def go_to_dft_domain(img):
    out = fftshift(fft2(img))
    return out

def makeGaussianFilter(numRows, numCols, sigma):
    centerI = int(numRows / 2) + numRows % 2
    centerJ = int(numCols / 2) + numCols % 2
    gaussian = np.zeros([numRows, numCols])
    for i in range(numRows):
        for j in range(numCols):
            gaussian[i, j] = math.exp(-1.0 * ((i - centerI) ** 2 + (j - centerJ) ** 2) / (2 * sigma ** 2))
    return gaussian

def image_filter(img, filter):
    img_dft = fftshift(fft2(img))
    out_dft = img_dft * filter
    return out_dft

def cutoff(filter, cut_off, ishighpass):
    numRows, numCols = filter.shape[0], filter.shape[1]
    centerI = int(numRows / 2) + numRows % 2
    centerJ = int(numCols  / 2) + numCols  % 2
    for i in range(numRows):
        for j in range(numCols):
            if ishighpass:
                if ((i-centerI) ** 2 + (j-centerJ) ** 2)**0.5 < cut_off:
                    filter[i, j] = 0
            else:
                if ((i - centerI) ** 2 + (j - centerJ) ** 2) ** 0.5 > cut_off:
                    filter[i, j] = 0
    return filter

image_near = cv2.imread('q4_01_near.jpg')
image_far = cv2.imread('q4_02_far.jpg')
image_near = cv2.resize(image_near, (image_far.shape[1], image_far.shape[0]), interpolation= cv2.INTER_AREA)

# x1_S = np.array([222, 248, 1])
# x2_S = np.array([222, 451, 1])
# x3_S = np.array([362, 348, 1])
# x4_S = np.array([437, 350, 1])
#
# x1_D = np.array([198, 235, 1])
# x2_D = np.array([198, 438, 1])
# x3_D = np.array([338, 335, 1])
# x4_D = np.array([413, 336, 1])
# point_S = np.array([x1_S, x2_S, x3_S, x4_S])
# point_D = np.array([x1_D, x2_D, x3_D, x4_D])
# h = cv2.findHomography(point_S, point_D)[0]
# print(h)
# image_far = warp(image_far, h, [image_far.shape[0], image_far.shape[1]])

cv2.imwrite('q4_03_near.jpg', image_near)
cv2.imwrite('q4_04_far.jpg', image_far)

image_near = cv2.imread('q4_03_near.jpg', 0)
image_near_dft = go_to_dft_domain(image_near)
amplitude_image_near = np.abs(image_near_dft)
log_amplitude_image_near = np.log(amplitude_image_near)
log_amplitude_image_near = log_amplitude_image_near - np.min(log_amplitude_image_near)
log_amplitude_image_near = 255*log_amplitude_image_near/np.max(log_amplitude_image_near)
cv2.imwrite('q4_05_dft_near.jpg', log_amplitude_image_near)

image_far = cv2.imread('q4_04_far.jpg', 0)
image_far_dft = go_to_dft_domain(image_far)
amplitude_image_far = np.abs(image_far_dft)
log_amplitude_image_far = np.log(amplitude_image_far)
log_amplitude_image_far = log_amplitude_image_far - np.min(log_amplitude_image_far)
log_amplitude_image_far = 255*log_amplitude_image_far/np.max(log_amplitude_image_far)
cv2.imwrite('q4_06_dft_far.jpg', log_amplitude_image_far)

low_pass = makeGaussianFilter(image_far.shape[0], image_far.shape[1], 10)
high_pass = 1 - makeGaussianFilter(image_near.shape[0], image_near.shape[1], 25)

l = low_pass
l = l - np.min(l)
l = 255*l/np.max(l)
cv2.imwrite('Q4_08_lowpass_10.jpg', l)

u = high_pass
u = u - np.min(u)
u = 255*u/np.max(u)
cv2.imwrite('Q4_07_highpass_25.jpg', u)

low_pass = cutoff(low_pass, 30, ishighpass= 0)
high_pass = cutoff(high_pass, 10, ishighpass= 1)

l = low_pass
l = l - np.min(l)
l = 255*l/np.max(l)
cv2.imwrite('Q4_10_lowpass_cutoff.jpg', l)

u = high_pass
u = u - np.min(u)
u = 255*u/np.max(u)
cv2.imwrite('Q4_09_highpass_cutoff.jpg', u)

img_near = cv2.imread('q4_03_near.jpg')
img_far = cv2.imread('q4_04_far.jpg')

img_lowpass_filtered = np.zeros(img_far.shape)
img_highpass_filtered = np.zeros(img_near.shape)
out_in_frequency_domain = np.zeros(img_near.shape)
out = np.zeros(img_near.shape)
out_far = cv2.resize(out, (int(out.shape[1]/10), int(out.shape[0]/10)), interpolation= cv2.INTER_AREA)

for i in range(3):
    image_near = img_near[:, :, i]
    image_far = img_far[:, :, i]
    a = image_filter(image_far, low_pass)
    b = image_filter(image_near, high_pass)
    c = 0.35 * a + 0.65 * b
    img_lowpass_filtered[:, :, i] = np.abs(a)
    img_highpass_filtered[:, :, i] = np.abs(b)
    out_in_frequency_domain[:, :, i] = np.abs(c)
    A, B = np.abs(ifft2(ifftshift(a))), np.abs(ifft2(ifftshift(b)))
    out[:, :, i] = 0.35*A + 0.65*B
    out_far[:, :, i] = cv2.resize(out[:, :, i], (int(out.shape[1]/10), int(out.shape[0]/10)), interpolation= cv2.INTER_AREA)

cv2.imwrite('Q4_11_highpassed.jpg', np.abs(img_highpass_filtered))
cv2.imwrite('Q4_12_lowpassed.jpg', np.abs(img_lowpass_filtered))

cv2.imwrite('Q4_13_hybrid_frequency.jpg', out_in_frequency_domain)

out = out - np.min(out)
out = 255 * out / np.max(out)
# out = cv2.resize(out, (out.shape[1]*2, out.shape[0]*2), interpolation= cv2.INTER_AREA)
cv2.imwrite('Q4_14_hybrid_near.jpg', out)

out_far = out_far - np.min(out_far)
out_far = 255 * out_far / np.max(out_far)
cv2.imwrite('Q4_15_hybrid_far.jpg', out_far)
