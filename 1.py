import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
os.add_dll_directory(r'D:\c12_p311\bin')
import cv2
import numpy as np

H, W = 400, 600
merged_image = np.full((H, W), 200, dtype=np.uint8)  # 灰色背景
cv2.putText(merged_image, "BACKGROUND", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 50, 3)

# -----------------------------
# 2. 生成前景图 (单通道灰度)
# -----------------------------
h1, w1 = 100, 150
moving_img_end = np.full((h1, w1), 200, dtype=np.uint8)  # 深灰色块
cv2.circle(moving_img_end, (w1//2, h1//2), 40, 200, -1)  # 白色圆

# -----------------------------
# 3. 指定放置位置
# -----------------------------
start_x, start_y = 200, 150

# -----------------------------
# 4. 创建掩码
# -----------------------------
mask = 255 * np.ones((h1, w1), dtype=np.uint8)

# -----------------------------
# 5. 计算中心点
# -----------------------------
center = (start_x + w1//2, start_y + h1//2)

# -----------------------------
# 6. 调用 seamlessClone
# -----------------------------
merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2BGR)
moving_img_end_bgr = cv2.cvtColor(moving_img_end, cv2.COLOR_GRAY2BGR)

output = cv2.seamlessClone(
    moving_img_end_bgr,
    merged_image_bgr,
    mask,
    center,
    cv2.NORMAL_CLONE
)

# -----------------------------
# 7. 显示和保存结果
# -----------------------------
cv2.imshow("Background", merged_image)
cv2.imshow("Foreground", moving_img_end)
cv2.imshow("SeamlessClone", output)
cv2.imwrite("output_test_gray.png", output)
cv2.waitKey(0)
cv2.destroyAllWindows()