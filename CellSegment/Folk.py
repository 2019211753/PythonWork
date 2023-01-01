import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal
import math

img = cv2.imread('default.jpg', 0)

# step1 阈值法图像分割
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.imshow(thresh, cmap='gray')
plt.suptitle('fixed threshold')
plt.show()

# step2 Sobel算子实现边缘检测
pho = np.array(Image.open('./data/test/0.png').convert('L'))

gate = 0.7 * 255

sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

x = signal.convolve2d(pho, sobel_x, 'same')
y = signal.convolve2d(pho, sobel_y, 'same')
grad = np.sqrt(np.square(x) + np.square(y))

dim = np.shape(grad)
for i in range(dim[0]):
    for j in range(dim[1]):
        if grad[i, j] < gate:
            pho[i, j] = 0

plt.imshow(pho, cmap='gray')
plt.suptitle('sobel')
plt.show()


# step3 滤波器
def Duv(i, j, P, Q):
    return math.sqrt(pow(i - P / 2, 2) + pow(j - Q / 2, 2))


# 布特沃斯高通滤波器
def BHPF(D0, P, Q, n):
    FIR = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            FIR[i, j] = 1 / (1 + math.pow(Duv(i, j, P, Q) / D0, 2 * n))

    return FIR


# 得到滤波后的频域图
def afterFilter():
    G = np.multiply(F, R)
    return G


# 傅里叶逆变换 得到原图
def restore(G):
    G0 = np.fft.ifft2(np.fft.fftshift(G))
    G0 = G0[1: M, 1: N]
    G0 = np.real(G0)
    plt.imshow(G0, cmap='gray')
    plt.title("5阶布特沃斯高通滤波器滤波后的图像")
    plt.show()


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# 计算填充图像大小
M, N = img.shape
P = 2 * M
Q = 2 * N

F = np.fft.fftshift(np.fft.fft2(pho, [P, Q]))
D0 = 200 # 截止频率
R = BHPF(D0, P, Q, 5)
G = afterFilter()
restore(G)
