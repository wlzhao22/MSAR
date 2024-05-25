import numpy as np
import matplotlib.pyplot as plt

def dfs(matrix, i, j, visited, region):
    """
    深度优先搜索函数
    """
    rows, cols = matrix.shape

    # 检查边界条件和已访问标记
    if (i < 0 or i >= rows or
            j < 0 or j >= cols or
            visited[i][j] or matrix[i][j] == 0):
        return

    # 标记当前位置已访问
    visited[i][j] = True

    # 添加当前位置到区域列表
    region.append(np.array([i, j]))

    # 递归探索相邻的四个方向
    dfs(matrix, i-1, j, visited, region)  # 上
    dfs(matrix, i+1, j, visited, region)  # 下
    dfs(matrix, i, j-1, visited, region)  # 左
    dfs(matrix, i, j+1, visited, region)  # 右


def find_regions(matrix):
    """
    查找所有区域的函数
    """
    rows, cols = matrix.shape

    visited = np.zeros((rows, cols), dtype=bool)  # 记录已访问的位置
    regions = []  # 存储所有区域的列表

    # 遍历矩阵中的每个位置
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and matrix[i][j] == 1:
                region = []  # 存储当前区域的坐标列表
                dfs(matrix, i, j, visited, region)  # 进行深度优先搜索
                regions.append(region)  # 将区域添加到列表中

    return regions

def show_img_and_box(img, boxes, title=None, axis=False):
    colors = ['r', 'g', 'b', 'y', 'm']
    fig, ax = plt.subplots()
    ax.imshow(img)
    print(img.shape)
    for i in range(len(boxes)):
        ax.add_patch(
            plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0],
                          boxes[i][3] - boxes[i][1], fill=False,
                          edgecolor=colors[i % 5], linewidth=3))
    print(title)
    if title is not None:
        print(title)
        plt.title(title)
    if not axis:
        plt.axis("off")
    plt.show()