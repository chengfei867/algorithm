"""
                        禁忌搜索算法
使用禁忌搜索算法解决商旅路径问题(TSP问题)。
初始化: 给定30个城市的坐标,计算这30个城市两两之间的距离.在画布上画出这30个城市

步骤：
        1、随机生成一个初始解，初始化禁忌表为空
        2、根据当前解使用某一规则生成一组新解，从新解中选出最优解
        3、若最优解不在禁忌表内，则将该解更新为当前最优解；若最优解在禁忌表内，则进行如下判断：
            3.1 若该解优于当前总体最优解 则促发破禁规则，接受该解；
            3.2 否则从新解中选取次优解
        4、更新新解为当前最优解，并将该解入队
        5、循环执行2~4，直到达到终止条件
"""

import math
from collections import deque
import random

# 初始化
## 定义30个城市的横纵坐标
city_num = 30
position_x = [
    24, 37, 54, 25, 7, 2, 68, 70, 54, 83, 64, 18, 22, 83, 21, 25, 24, 58, 71, 74, 87, 18, 13, 82, 62, 58, 45, 41, 44,
    42]
position_y = [
    44, 84, 67, 62, 64, 79, 58, 44, 62, 69, 60, 54, 60, 46, 58, 68, 42, 69, 71, 78, 76, 40, 40, 7, 32, 35, 21, 26, 35,
    20]

## 定义城市间的距离矩阵
distance_city = [[0 for _ in range(city_num)] for _ in range(city_num)]
## 计算城市间两两的距离
for i in range(city_num):
    for j in range(city_num):
        distance_city[i][j] = int(
            math.sqrt((position_x[i] - position_x[j]) ** 2 + (position_y[i] - position_y[j]) ** 2))

# 主函数
def main():
    # 创建一个最大容量为100的禁忌表
    tabu_list = deque(maxlen=30)
    # 获得初始解
    init_path = getPathRandomly()
    best_path = init_path
    best_distance = getDistanceByPath(best_path)
    current_path = init_path
    # 循环执行
    for iteration in range(10000):
        # 根据当前解生成一组新解
        new_paths_dict = getNearlyPaths(current_path)
        # 循环直到找到一个满足要求的当前解
        while 1:
            # 从new_paths_dict中选择最优路径
            current_best_path, distance, city_index = getBestPath(new_paths_dict)
            # 判断当前得到当前最优解是否是目前全局最优解
            if distance < best_distance:
                # 是，则更新当前解以及全局最优解
                current_path = current_best_path
                best_path = current_best_path
                best_distance = distance
                # 将city_index加入禁忌表
                tabu_list.append(city_index)
                break
            # 虽然不是当前全局最优解，但是其操作不在禁忌表中，依然选择该解
            elif city_index not in tabu_list:
                current_path = current_best_path
                # 将city_index加入禁忌表
                tabu_list.append(city_index)
                break
            # 不是当前全局最优解，并且操作还在禁忌表中，则选择当前次优解
            else:
                # 从new_paths_dict中删除最优解
                del new_paths_dict[tuple(current_best_path)]
    print(f"最终路径为:{current_path},最终距离为:{getDistanceByPath(current_path)}")
    print(f"最优路径为:{best_path},最短距离为:{best_distance}")
    return current_path


def getPathRandomly():
    # 找到的新解的路径
    path = []
    # 每次访问一个城市就将其编号从中删除其中
    accessed = [i for i in range(city_num)]
    # 生成随机初始出发城市
    start_city = random.randint(0, city_num - 1)
    path.append(start_city)  # 将初始城市加入路径列表
    accessed.remove(start_city)  # 将初始城市加入访问列表
    # 循环直到所有城市都遍历到
    while len(accessed) != 0:
        new_city = random.choice(accessed)  # 从未访问列表中随机选择一个城市
        path.append(new_city)  # 将新城市加入路径列表
        accessed.remove(new_city)  # 更新访问集合
    # print(f"初始解为：{path}")
    return path


# 根据当前解生成一组邻近解
def getNearlyPaths(current_path):
    new_paths_dict = {}
    for _ in range(200):
        new_path, city_index = swapRandomly(current_path)
        # print(f"new_path:{new_path},类型：{type(new_path)}")
        # print(f"city_index:{city_index},类型：{type(city_index)}")
        new_paths_dict[tuple(new_path)] = city_index
    return new_paths_dict

def swapRandomly(current_path):
    new_path = current_path.copy()
    # print(f"原路径{currentPath}")
    # 随机生成两个城市索引
    city_index = random.sample(range(city_num), 2)
    # 交换原路径处于这两个索引位置的城市编号
    new_path[city_index[0]], new_path[city_index[1]] = new_path[city_index[1]], new_path[city_index[0]]
    # print(f"新路径{current_path}")
    return new_path, city_index


def getBestPath(new_paths_dict):
    # 获取路径列表
    new_paths = [list(t) for t in new_paths_dict.keys()]
    # 初始化最优路径
    best_path = new_paths[0]
    # 初始化最短距离
    best_distance = getDistanceByPath(best_path)
    # 遍历路径列表
    for new_path in new_paths:
        new_path_distance = getDistanceByPath(new_path)
        if new_path_distance < best_distance:
            best_path = new_path
            best_distance = new_path_distance
    # 返回最优路径的操作
    city_index = new_paths_dict[tuple(best_path)]
    return best_path, best_distance, city_index


def getDistanceByPath(path):
    # 初始化距离0
    distance = 0
    # 计算距离
    for i in range(len(path) - 1):
        distance += distance_city[path[i]][path[i + 1]]
    distance += distance_city[path[0]][path[city_num - 1]]
    return distance

# 入口函数
if __name__ == "__main__":
    main()
