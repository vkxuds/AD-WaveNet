import torch
import torch.nn.functional as F
import numpy as np
import cv2
#import matplotlib.pyplot as plt

def gaussian_kernel(size: int, sigma: float):
    """Returns a 2D Gaussian kernel."""
    kernel_1d = torch.linspace(-(size // 2), size // 2, size)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = torch.exp(-0.5 * (kernel_2d ** 2) / sigma ** 2)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d

def conv2d(image, kernel):
    """Performs 2D convolution using PyTorch."""
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return F.conv2d(image, kernel, padding=kernel.size(-1) // 2).squeeze()
def sobel_kernels(device):
    """Returns Sobel kernels for x and y gradients."""
    Kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).to(device)
    Ky = torch.tensor([[ 1., 2., 1.], [ 0., 0., 0.], [-1., -2., -1.]]).to(device)
    return Kx, Ky

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """Performs non-maximum suppression on the gradient magnitudes."""
    H, W = gradient_magnitude.shape
    Z = torch.zeros((H, W), dtype=torch.float32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            try:
                q = 1.0
                r = 1.0
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    Z[i, j] = gradient_magnitude[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

def threshold(image, low, high):
    """Applies double threshold to the image."""
    high_threshold = image.max() * high
    low_threshold = high_threshold * low

    M, N = image.shape
    res = torch.zeros((M, N), dtype=torch.float32)

    strong = 1.0
    weak = 0.5

    strong_i, strong_j = torch.where(image >= high_threshold)
    zeros_i, zeros_j = torch.where(image < low_threshold)

    weak_i, weak_j = torch.where((image <= high_threshold) & (image >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

def hysteresis(image, weak, strong=1.0):
    """Applies hysteresis to the thresholded image."""
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if image[i, j] == weak:
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                    or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                    or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image, low_threshold=0.001, high_threshold=0.01):
    """Performs Canny edge detection using PyTorch."""
    # Step 1: Noise reduction (Gaussian filter)
    #gauss_kernel = gaussian_kernel(size=5, sigma=1.4).to(image.device)
    #smoothed = conv2d(image, gauss_kernel).to(image.device)

    # Step 2: Finding intensity gradient (Sobel filter)
    Kx, Ky = sobel_kernels(image.device)
    Ix = conv2d(image, Kx)
    Iy = conv2d(image, Ky)

    gradient_magnitude = torch.sqrt(Ix**2 + Iy**2)
    gradient_direction = torch.atan2(Iy, Ix)

    # Step 3: Non-maximum suppression
    #non_max_suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Step 4: Double threshold
    #thresholded, weak, strong = threshold(non_max_suppressed, low_threshold, high_threshold)

    # Step 5: Edge tracking by hysteresis
    #edges = hysteresis(thresholded, weak, strong)

    return gradient_direction

'''# 读取图像并转换为灰度图
image_path = '0_1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = torch.tensor(image, dtype=torch.float32) / 255.0  # 归一化

# 执行Canny边缘检测
edges = canny_edge_detection(image, low_threshold=50/255, high_threshold=100/255)

# 显示原图和边缘检测结果
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image.numpy(), cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges.numpy(), cmap='gray'), plt.title('Canny Edges')
plt.show()
'''