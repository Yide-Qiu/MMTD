import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

matrix = np.random.rand(5, 5)
normalized_matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
# normalized_matrix[0, 2] = 0.8
# normalized_matrix[2, 0] = 0.8
for i in range(5):
    normalized_matrix[i, i] = 1.0
    for j in range(i, 5):
        normalized_matrix[i, j] = normalized_matrix[j, i]

print(normalized_matrix)

high_res_matrix = zoom(normalized_matrix, (128 / 5, 128 / 5), order=3)

# plt.imshow(high_res_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Association Estimating Probability')
# # plt.title('128x128 Heatmap of 5x5 Normalized Random Matrix')

plt.imshow(high_res_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Association Estimating Probability')
# plt.title('128x128 Heatmap of 5x5 Normalized Random Matrix')

num_labels = 5
tick_positions = np.linspace(0, 127, num_labels)  # 128像素对应5个标签
plt.xticks(tick_positions, range(1, num_labels + 1))  # 横轴标签为 1 到 5
plt.yticks(tick_positions, range(1, num_labels + 1))  # 纵轴标签为 1 到 5


plt.savefig('heatmap_128x128.png', dpi=300)
# plt.show()
