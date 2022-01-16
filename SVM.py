import cv2
import numpy as np
from skimage import feature as ft

SZ = 20
bin_n = 16

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


# 使用图片的二阶矩对图片进行抗扭斜处理
def deskew(img):
    m = cv2.moments(img)  # 获取图片的矩
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


# 计算图像的hog描述符
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def newhog(img):
    features = ft.hog(img,  # input image
                      orientations=9,  # number of bins
                      pixels_per_cell=4,  # pixel per cell
                      cells_per_block=4,  # cells per blcok
                      block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                      transform_sqrt=True,  # power law compression (also known as gamma correction)
                      feature_vector=True,  # flatten the final vectors
                        )  # return HOG map
    return features

# 获取图片，拆分图片
img = cv2.imread('digits.png', 0)
# 将数据分割成50行，100列
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

train_cells = [i[:50] for i in cells]  # 前50列作为训练数据
test_cells = [i[50:] for i in cells]  # 后50列作为测试数据

deskewed = [list(map(deskew, row)) for row in train_cells]  # 对训练数据进行抗扭斜数据
hogdata = [list(map(hog, row)) for row in deskewed]  # 获取数据的hog符
trainData = np.float32(hogdata).reshape(-1, 64)  # 行数不知道，-1表示自动计算，转变成64列

labels = np.repeat(np.arange(10), 250)[:, np.newaxis]  # 获取标签，

# 创建SVM分类器，设置各项参数
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)
svm.save('svm_data.dat')

# 对测试数据进行处理
deskewed = [list(map(deskew, row)) for row in test_cells]      # 对训练数据进行抗扭斜数据
hogdata = [list(map(hog, row)) for row in deskewed]            # 获取数据的hog符
testData = np.float32(hogdata).reshape(-1, bin_n * 4)

ret, result = svm.predict(testData)
print(result[0])
mask = result == labels
correct = np.count_nonzero(mask)
print(correct * 100.0 / len(result), '%')
print(len(result))
print(correct)

lower_red = np.array([0, 0, 0])
upper_red = np.array([160, 160, 160])    # 设置阈值下限和上限，去除背景颜色

img2 = cv2.imread('1.png')
cv2.imshow('ori',img2)
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv,lower_red,upper_red)
cv2.imshow('mask1',mask1)
cell2 = [np.hsplit(row, 2) for row in np.vsplit(mask1, 1)]
#deskewed2 = [list(map(deskew, row)) for row in cell2]
Myhogdata = [list(map(hog, row)) for row in cell2]
MyTest = np.float32(Myhogdata).reshape(-1, 64)
ret2, result2 = svm.predict(MyTest)
print(result2)

k = cv2.waitKey(50000)