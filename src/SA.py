'''
                        模拟退火算法
使用模拟退火算法解决商旅路径问题(TSP问题)。
初始化: 给定30个城市的坐标,计算这30个城市两两之间的距离.在画布上画出这30个城市
        设置初始温度T0,迭代次数k,降温幅度系数,生成一个初始解x0,并且算出E(x0)
步骤：
        1、产生一个新解x1,计算E(x1),计算ΔE=E(1)-E(0)
        2、若ΔE<0(表示当前路径比上次的短)则接受新解,否则以exp(-ΔE/T)的概率接受该解。
        3、更新当前最优解为新解
        4、循环1~3k次
        5、判断当前温度是否达到终止温度,是则终止,否则降低温度T,并且重置循环次数k。
'''

import math
import random
import matplotlib.pyplot as plt

itor = 0
bestPath, bestDistance = [], 0
# 需要更新的全局变量
x_coords = []
y_coords = []
fig, ax = plt.subplots()
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
    global itor
    global bestPath, bestDistance
    ## 设置初始参数
    T = 1000.0  # 初始温度
    ALPHA = 0.98  # 降温幅度系数
    # 初始解
    currentPath = getPathRandomly()
    # currentPath = list(range(30))
    currentDistance = getDistanceByPath(currentPath)
    # 最优解
    bestPath, bestDistance = currentPath, currentDistance
    # 进入循环
    while T > 0.1:
        K = 1000
        while K > 0:
            # 新解
            itor += 1
            path_new, distance_new = getPathBaseOnPath(currentPath)
            # 是否接受新解
            flag = isAccept(currentDistance, distance_new, T)
            # print(f"新解是否被接受{flag}")
            if flag:
                currentPath = path_new
                currentDistance = distance_new
                if currentDistance < bestDistance:
                    bestPath, bestDistance = currentPath, currentDistance
            K -= 1
        # 更新全局变量，用于动态展示
        global x_coords, y_coords
        x_coords = []
        y_coords = []
        for city_index in currentPath:
            x_coords.append(position_x[city_index])
            y_coords.append(position_y[city_index])
        x_coords.append(position_x[currentPath[0]])
        y_coords.append(position_y[currentPath[0]])

        # 绘制当前路径
        ax.clear()
        ax.plot(x_coords, y_coords, 'bo-')
        ax.set_title(f'Iteration {itor}')
        # 显示当前总路径长度
        ax.text(0.5, -0.1, f'Current Total Distance: {currentDistance:.2f}', transform=ax.transAxes, ha='center')
        plt.pause(0.01)  # 暂停一小段时间，用于展示动态效果
        # 更新当前温度
        # print(f"本伦温度为:{T}")
        # print(f"本轮最优路径为:{path},最短距离为:{distance}")
        T *= ALPHA

    print(f"总迭代次数{itor}")
    print(f"最终温度为:{T}")
    print(f"最终路径为:{currentPath},最短距离为:{currentDistance}")
    print(f"最优路径为:{bestPath},最短距离为:{bestDistance}")


# 随机产生新解(可优化)
def getPathRandomly():
    # 找到的新解的路径
    path = []
    # 每次访问一个城市就将其编号从中删除其中
    accessed = [i for i in range(city_num)]
    # 生成随机初始出发城市
    start_city = random.randint(0, city_num-1)
    path.append(start_city)  # 将初始城市加入路径列表
    accessed.remove(start_city)  # 将初始城市加入访问列表
    # 循环直到所有城市都遍历到
    while len(accessed) != 0:
        new_city = random.choice(accessed)  # 从未访问列表中随机选择一个城市
        path.append(new_city)  # 将新城市加入路径列表
        accessed.remove(new_city)  # 更新访问集合
    return path


# 在当前最优解的基础上生成新解
def getPathBaseOnPath(currentPath):
    # 找到的新解的距离
    distance = 0
    # 1 随机交换两个城市的位置
    # newPath = swapRandomly(currentPath)
    # 2 部分反转
    # newPath = partialInversion(currentPath)
    # 3 随机插入
    # newPath = insertRandomly(currentPath)
    index = random.randint(0, 2)
    if index == 0:
        newPath = swapRandomly(currentPath)
    elif index == 1:
        newPath = partialInversion(currentPath)
    else:
        newPath = insertRandomly(currentPath)
    distance = getDistanceByPath(newPath)
    # print(f"原路径:{currentPath}")
    # print(f"新路径:{newPath}")
    # 更新全局变量
    global x_coords, y_coords
    x_coords = []
    y_coords = []
    for city_index in newPath:
        x_coords.append(position_x[city_index])
        y_coords.append(position_y[city_index])
    x_coords.append(position_x[newPath[0]])
    y_coords.append(position_y[newPath[0]])
    return (newPath, distance)


# 随机交换两个城市的位置
def swapRandomly(currentPath):
    newPath = currentPath.copy()
    # print(f"原路径{currentPath}")
    # 随机生成两个城市索引
    city_index = random.sample(range(city_num), 2)
    # 交换原路径处于这两个索引位置的城市编号
    newPath[city_index[0]], newPath[city_index[1]] = newPath[city_index[1]], newPath[city_index[0]]
    # print(f"新路径{currentPath}")
    return newPath


# 部分反转
def partialInversion(currentPath):
    newPath = currentPath.copy()
    # print(f"原路径{currentPath}")
    # 随机生成两个城市索引
    city_index = random.sample(range(city_num), 2)
    # 确保start_index小于end_index
    start_index, end_index = min(city_index[0], city_index[1]), max(city_index[0], city_index[1])
    # 反转
    newPath[start_index:end_index + 1] = reversed(newPath[start_index:end_index + 1])
    # print(f"新路径{currentPath}")
    return newPath


def insertRandomly(currentPath):
    newPath = currentPath.copy()
    # print(f"insertRandomly原路径{currentPath}")
    # 随机生成一个城市索引和插入位置
    city_index = random.sample(range(city_num), 2)
    # 删除并拿到第一个索引城市编号
    temp_city_index = newPath.pop(city_index[0])
    # 插入新位置
    newPath.insert(city_index[1] - 1, temp_city_index)
    # print(f"insertRandomly新路径{currentPath}")   
    return newPath


# 计算给定路径的总距离
def getDistanceByPath(path):
    # 初始化距离0
    distance = 0
    # 计算距离
    for i in range(len(path) - 1):
        distance += distance_city[path[i]][path[i + 1]]
    distance += distance_city[path[0]][path[city_num-1]]
    return distance


# 判断是否接受新解
def isAccept(oldDist, newDist, T):
    distGap = newDist - oldDist
    if distGap < 0 or random.random() < math.exp(-(distGap) / T):
        return True
    return False

def plot_points(path):
    x_coords = []
    y_coords = []
    # 解析路径
    for city_index in path:
        x_coords.append(position_x[city_index])
        y_coords.append(position_y[city_index])
    x_coords.append(position_x[path[0]])
    y_coords.append(position_y[path[0]])
    # print(f"xcoords:{x_coords}")
    # print(f"ycoords:{y_coords}")

    # 创建图
    plt.figure()

    # 将点的坐标连接成线段
    plt.plot(x_coords, y_coords, 'bo-')
    plt.scatter(x_coords[0], y_coords[0], color='red', label='Start')

    # 设置坐标轴标签
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 显示图
    plt.show()


# 入口函数
if __name__ == "__main__":
    main()
    # 最终展示最优路径
    plot_points(bestPath)
