import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_variance(image_gray):
    """
    计算图像的拉普拉斯方差（Laplacian Variance），数值越大，图像越清晰。
    """
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    return lap.var()

def tenengrad(image_gray, ksize=3):
    """
    通过 Sobel 算子计算图像的 Tenengrad 值，反映梯度（边缘）信息，数值越大越清晰。
    """
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_square = gx**2 + gy**2
    return np.mean(grad_square)

def combined_focus_score(image_gray, alpha=1.0, beta=1.0):
    """
    组合拉普拉斯方差和 Tenengrad 得分
    得分公式： score = alpha * Laplacian Variance + beta * Tenengrad Score
    """
    score_lap = laplacian_variance(image_gray)
    score_ten = tenengrad(image_gray)
    return alpha * score_lap + beta * score_ten

def find_best_focused_image(image_stack, alpha=1.0, beta=1.0, roi=None):
    """
    对已存在的图像堆栈（列表或 ndarray）进行逐张分析，选出焦点得分最高的图像。
    
    参数：
      image_stack: 图像列表（每个图像可以是灰度或彩色）
      alpha, beta: 组合焦点评分时使用的权重
      roi: 可选，(x, y, w, h) 限定的感兴趣区域；若为 None，则在全图上计算

    返回：
      best_index: 得分最高的图像在列表中的索引
      best_score: 最佳图像的焦点得分
      best_img  : 焦点最优的图像
      scores    : 每一幅图像对应的焦点得分列表
    """
    best_score = -1
    best_index = -1
    best_img = None
    scores = []

    for i, img in enumerate(image_stack):
        # 如果是彩色图像，则转换为灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 如果指定了 ROI，则只计算 ROI 部分
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # 计算综合焦点得分
        score = combined_focus_score(gray, alpha, beta)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_index = i
            best_img = img

    return best_index, best_score, best_img, scores

def generate_synthetic_images(image_size=(512, 512), num_images=5):
    """
    生成一组合成图像：
      - 先生成一张包含网格线的原始清晰图像（模拟理想聚焦图像）。
      - 对原图按不同程度进行高斯模糊，模拟不同焦平面下采集的图像堆栈。
    """
    # 构造一张空白图像并绘制网格
    sharp_image = np.zeros(image_size, dtype=np.uint8)
    for i in range(0, image_size[0], 20):
        cv2.line(sharp_image, (0, i), (image_size[1], i), color=255, thickness=1)
    for j in range(0, image_size[1], 20):
        cv2.line(sharp_image, (j, 0), (j, image_size[0]), color=255, thickness=1)
    
    # 根据不同的高斯核大小，模拟不同程度的模糊
    image_stack = []
    # 当核大小为 1 时，相当于无模糊，即最佳对焦图像
    blur_kernel_sizes = [1, 3, 5, 7, 9]
    for ksize in blur_kernel_sizes:
        if ksize == 1:
            blurred = sharp_image.copy()
        else:
            blurred = cv2.GaussianBlur(sharp_image, (ksize, ksize), 0)
        image_stack.append(blurred)
    
    return image_stack

def main():
    # 生成一组模拟图像堆栈
    image_stack = generate_synthetic_images()

    # 在堆栈中计算每张图的焦点得分，并选出最佳图像
    best_index, best_score, best_img, scores = find_best_focused_image(image_stack, alpha=1.0, beta=1.0, roi=None)
    print(f"最佳焦点图像索引: {best_index}, 焦点得分: {best_score:.2f}")
    for idx, score in enumerate(scores):
        print(f"图像 {idx} 的焦点得分 = {score:.2f}")

    # 显示所有图像及对应得分
    plt.figure(figsize=(12, 4))
    for i, img in enumerate(image_stack):
        plt.subplot(1, len(image_stack), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"索引 {i}\n得分: {scores[i]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 单独显示最佳焦点图像
    plt.figure()
    plt.imshow(best_img, cmap='gray')
    plt.title(f"最佳焦点图像（索引 {best_index}）")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
