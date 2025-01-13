import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Tạo một tập dữ liệu mẫu (5 điểm dữ liệu không đều)
points = np.array([[0.1, 0.2], [0.4, 0.4], [0.7, 0.6], [0.9, 0.8], [0.3, 0.7]])
values = np.array([1, 2, 3, 4, 5])  # Giá trị tại các điểm

# Tạo lưới đều để nội suy
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]  # 100x100 lưới

# Nội suy với 3 phương pháp
grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_linear = griddata(points, values, (grid_x, grid_y), method='linear')
grid_cubic = griddata(points, values, (grid_x, grid_y), method='cubic')

# Vẽ kết quả
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Biểu đồ điểm gốc
axs[0].scatter(points[:, 0], points[:, 1], c=values, s=100, cmap='viridis', edgecolor='k')
axs[0].set_title('Original Data Points')
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)

# Nội suy nearest
im_nearest = axs[1].imshow(grid_nearest.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
axs[1].scatter(points[:, 0], points[:, 1], c=values, edgecolor='k')
axs[1].set_title('Nearest Interpolation')
plt.colorbar(im_nearest, ax=axs[1])

# Nội suy linear
im_linear = axs[2].imshow(grid_linear.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
axs[2].scatter(points[:, 0], points[:, 1], c=values, edgecolor='k')
axs[2].set_title('Linear Interpolation')
plt.colorbar(im_linear, ax=axs[2])

# Nội suy cubic
im_cubic = axs[3].imshow(grid_cubic.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
axs[3].scatter(points[:, 0], points[:, 1], c=values, edgecolor='k')
axs[3].set_title('Cubic Interpolation')
plt.colorbar(im_cubic, ax=axs[3])

plt.tight_layout()
plt.show()
