import numpy as np
import itertools
import random
from collections import defaultdict
import csv
import time
import os
import graphviz
import platform
import subprocess
import sys
import shutil

# 设置随机种子确保可重复性
random.seed(42)
np.random.seed(42)


def check_graphviz_installation():
    """检查Graphviz是否已正确安装并配置"""
    # 检查dot命令是否可用
    dot_path = shutil.which('dot')
    if dot_path:
        print(f"Graphviz 已安装: {dot_path}")
        return True

    # 检查常见安装路径
    common_paths = [
        r"C:\Program Files\Graphviz\bin\dot.exe",
        r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
        r"C:\Program Files\Graphviz\bin\dot",
        r"C:\Program Files (x86)\Graphviz\bin\dot"
    ]

    for path in common_paths:
        if os.path.exists(path):
            # 将路径添加到系统环境变量
            bin_dir = os.path.dirname(path)
            os.environ["PATH"] += os.pathsep + bin_dir
            print(f"发现 Graphviz: {path}")
            return True

    return False


class MatrixGenerator:
    """邻接矩阵生成器（实现论文第5节的方法）"""

    def __init__(self):
        # 使用随机实数作为变量值（避免整数冲突）
        self.vars = {
            'c': random.uniform(1.5, 2.5),  # 载体对角线值
            'p': random.uniform(2.5, 3.5),  # 行星对角线值
            's': random.uniform(3.5, 4.5),  # 太阳轮对角线值
            'r': random.uniform(4.5, 5.5),  # 转动副值
            'g': random.uniform(5.5, 6.5),  # 齿轮副值
            '1': 1.0,  # 多重转动副值
        }

    def evaluate(self, expr):
        """计算表达式的值"""
        if isinstance(expr, (int, float)):
            return expr
        return self.vars.get(expr, 0.0)

    def generate_A_matrix(self, n_c, n_p, n_s, turning_partition, gear_edges, sun_edges):
        """
        生成完整图的邻接矩阵A（如论文5.2节）

        参数:
            n_c: 载体数量
            n_p: 行星数量
            n_s: 太阳轮数量
            turning_partition: 转动副分区 (行星分配方案)
            gear_edges: 载体-行星齿轮副连接列表 [(载体索引, 行星索引)]
            sun_edges: 太阳轮-行星齿轮副连接列表 [(太阳轮索引, 行星索引)]
        """
        size = n_c + n_p + n_s
        matrix = np.zeros((size, size), dtype=object)

        # 节点索引分配:
        # 0 到 n_c-1: 载体
        # n_c 到 n_c+n_p-1: 行星
        # n_c+n_p 到 n_c+n_p+n_s-1: 太阳轮

        # 设置对角线元素
        for i in range(n_c):
            matrix[i, i] = 'c'  # 载体
        for i in range(n_c, n_c + n_p):
            matrix[i, i] = 'p'  # 行星
        for i in range(n_c + n_p, size):
            matrix[i, i] = 's'  # 太阳轮

        # 设置载体之间的多重转动副（不再表示）
        # 设置太阳轮之间的多重转动副（不再表示）

        # 设置行星与载体之间的转动副
        planet_offset = 0
        for carrier_idx, planet_count in enumerate(turning_partition):
            for j in range(planet_count):
                planet_idx = n_c + planet_offset + j


                matrix[carrier_idx, planet_idx] = 'r'
                matrix[planet_idx, carrier_idx] = 'r'
            planet_offset += planet_count

        # 设置行星与载体之间的齿轮副
        for carrier_idx, planet_idx in gear_edges:
            planet_global_idx = n_c + planet_idx
            matrix[carrier_idx, planet_global_idx] = 'g'
            matrix[planet_global_idx, carrier_idx] = 'g'

        # 设置行星与太阳轮之间的齿轮副
        for sun_idx, planet_idx in sun_edges:
            sun_global_idx = n_c + n_p + sun_idx
            planet_global_idx = n_c + planet_idx
            matrix[sun_global_idx, planet_global_idx] = 'g'
            matrix[planet_global_idx, sun_global_idx] = 'g'







        # 将None替换为0
        matrix[matrix == None] = 0

        return matrix

    def compute_determinant(self, matrix):
        """计算矩阵的行列式（数值计算）"""
        num_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                num_matrix[i, j] = self.evaluate(matrix[i, j])
        return np.linalg.det(num_matrix)


def integer_partitions(n, k):
    """生成n划分为k个正整数的所有唯一分区（已排序）"""
    if k == 0:
        return [()] if n == 0 else []
    if k == 1:
        return [(n,)]

    partitions = []
    for first in range(1, n - k + 2):
        for rest in integer_partitions(n - first, k - 1):
            # 创建排序后的分区
            partition = (first,) + rest
            partitions.append(partition)

    # 返回唯一分区（已排序）
    return sorted(set(partitions))


def get_link_combinations(n_min=4, n_max=20):
    """生成所有有效的连杆组合（n=4到20）"""
    combinations = defaultdict(list)

    for n in range(n_min, n_max + 1):
        # 根据论文中的约束条件生成组合
        for n_c in range(0, n + 1):
            for n_p in range(0, n + 1 - n_c):
                n_s = n - n_c - n_p

                # 约束1: n_s + n_c >= 3
                if n_s + n_c < 3:
                    continue

                # 约束2: 2 * n_p <= n - 2
                if 2 * n_p > n - 2:
                    continue

                # 约束3: 2 * n_p >= n - 1 - 2 * n_c
                if 2 * n_p < n - 1 - 2 * n_c:
                    continue

                # 约束4: n_c <= n_p
                if n_c > n_p:
                    continue

                combinations[n].append((n_c, n_p, n_s))

    return combinations


def solve_gear_assignment(n_p, n_s, planet_assign, sun_assign):
    """
    解决齿轮分配问题

    参数:
        n_p: 行星数量
        n_s: 太阳轮数量
        planet_assign: 行星的齿轮副分配列表
        sun_assign: 太阳轮的齿轮副分配列表

    返回:
        所有有效的太阳轮-行星连接方案列表
    """
    # 如果行星或太阳轮数量为0，返回空列表
    if n_p == 0 or n_s == 0:
        return []

    # 行星和太阳轮之间的可能连接数量
    num_vars = n_p * n_s

    # 创建所有可能的连接矩阵（0/1）
    possible_matrices = []

    # 生成所有可能的连接组合
    for assignment in itertools.product([0, 1], repeat=num_vars):
        matrix = np.array(assignment).reshape(n_p, n_s)

        # 检查行星约束
        valid = True
        for i in range(n_p):
            if np.sum(matrix[i, :]) != planet_assign[i]:
                valid = False
                break

        if not valid:
            continue

        # 检查太阳轮约束
        for j in range(n_s):
            if np.sum(matrix[:, j]) != sun_assign[j]:
                valid = False
                break

        if valid:
            # 将矩阵转换为连接列表
            connections = []
            for i in range(n_p):
                for j in range(n_s):
                    if matrix[i, j] == 1:
                        connections.append((j, i))  # (太阳轮索引, 行星索引)
            possible_matrices.append(connections)

    return possible_matrices


def generate_complete_graphs(n_min=4, n_max=11):
    """生成完整的行星齿轮系图（n=4到6）"""
    matrix_gen = MatrixGenerator()
    link_combinations = get_link_combinations(n_min, n_max)
    complete_graphs = defaultdict(list)

    print(f"生成 {n_min} 到 {n_max} 构件的完整PGT图...")
    start_time = time.time()

    for n in range(n_min, n_max + 1):
        if n not in link_combinations:
            continue

        graph_count = 0
        for comb in link_combinations[n]:
            n_c, n_p, n_s = comb
            j = n - 2  # 总齿轮副数

            # Step 1: 行星-载体转动副子图
            if n_c > 0:
                partitions = integer_partitions(n_p, n_c)
            else:
                partitions = [()]

            # Step 2: 添加行星-载体齿轮副
            j_c_max = min(
                2 * (n_c + n_p) - n + 1,  # 公式(6)
                n_c + n_p - 2,
                n_p * (n_c - 1) if n_c > 1 else 0
            )
            j_c_max = max(0, j_c_max)  # 确保非负

            for turning_part in partitions:
                for j_c in range(0, j_c_max + 1):
                    # Step 3: 分配太阳轮-行星齿轮副
                    j_s = j - j_c
                    if j_s < 0:
                        continue

                    # 太阳轮分配
                    sun_assignments = []
                    if n_s > 0:
                        for assign in integer_partitions(j_s, n_s):
                            if sum(1 for a in assign if a == 1) <= 3 and all(a <= n_p for a in assign):
                                sun_assignments.append(assign)
                    else:
                        if j_s == 0:
                            sun_assignments = [()]

                    # 行星分配
                    planet_assignments = []
                    if n_p > 0:
                        for assign in integer_partitions(j_s, n_p):
                            if all(a <= n_s for a in assign):
                                planet_assignments.append(assign)
                    else:
                        if j_s == 0:
                            planet_assignments = [()]

                    if not sun_assignments or not planet_assignments:
                        continue

                    # 为行星-载体齿轮副生成所有可能的连接
                    possible_edges = []
                    planet_offset = 0
                    for carrier_idx, planet_count in enumerate(turning_part):
                        for local_idx in range(planet_count):
                            planet_idx = planet_offset + local_idx
                            for other_carrier in range(n_c):
                                if other_carrier != carrier_idx:
                                    possible_edges.append((other_carrier, planet_idx))
                        planet_offset += planet_count

                    # 枚举所有可能的齿轮副组合
                    edge_combinations = []
                    if possible_edges and j_c > 0:
                        if j_c <= len(possible_edges):
                            edge_combinations = itertools.combinations(possible_edges, j_c)
                        else:
                            continue
                    else:
                        edge_combinations = [[]]  # 没有齿轮副的情况

                    for gear_edges in edge_combinations:
                        # 处理太阳轮和行星分配
                        for sun_assign in sun_assignments:
                            for planet_assign in planet_assignments:
                                # 解决行星-太阳轮连接问题
                                sun_edge_solutions = solve_gear_assignment(n_p, n_s, planet_assign, sun_assign)

                                for sun_edges in sun_edge_solutions:
                                    # 构建完整邻接矩阵A
                                    A_matrix = matrix_gen.generate_A_matrix(
                                        n_c, n_p, n_s, turning_part, gear_edges, sun_edges)

                                    # 计算行列式签名
                                    det_signature = matrix_gen.compute_determinant(A_matrix)

                                    # 创建完整图表示
                                    graph = {
                                        "n": n,
                                        "n_c": n_c,
                                        "n_p": n_p,
                                        "n_s": n_s,
                                        "turning_partition": turning_part,
                                        "gear_edges": gear_edges,
                                        "sun_edges": sun_edges,
                                        "sun_assignment": sun_assign,
                                        "planet_assignment": planet_assign,
                                        "det_signature": det_signature
                                    }

                                    # 检查是否已存在同构图
                                    is_duplicate = False
                                    for existing in complete_graphs[n]:
                                        if abs(det_signature - existing["det_signature"]) < 1e-5:
                                            is_duplicate = True
                                            break

                                    if not is_duplicate:
                                        complete_graphs[n].append(graph)
                                        graph_count += 1

        print(f"n={n}: 生成 {graph_count} 个完整图")

    end_time = time.time()
    print(f"生成完成! 用时 {end_time - start_time:.2f} 秒")
    return complete_graphs


# def create_graphviz_diagram(graph):
#     """
#     使用graphviz创建PGT图的可视化 - 三层纵向布局
#
#     参数:
#         graph: 包含PGT图信息的字典
#
#     返回:
#         graphviz.Digraph对象
#     """
#     n_c = graph["n_c"]
#     n_p = graph["n_p"]
#     n_s = graph["n_s"]
#
#     # 创建有向图
#     dot = graphviz.Digraph(
#         format='png',
#         graph_attr={
#             'rankdir': 'TB',  # 从上到下的布局
#             'label': f'PGT Graph (n={graph["n"]})',
#             'labelloc': 't',
#             'fontsize': '16',
#             'nodesep': '0.5',  # 节点间距
#             'ranksep': '1.0'  # 层级间距
#         },
#         node_attr={
#             'shape': 'circle',  # 构件用圆圈表示
#             'style': 'filled',
#             'width': '0.6',
#             'height': '0.6',
#             'fixedsize': 'true'
#         },
#         edge_attr={
#             'arrowhead': 'none'  # 移除所有箭头
#         }
#     )
#
#     # 添加载体节点（顶层）
#     dot.attr('node', fillcolor='lightblue')
#     for i in range(n_c):
#         dot.node(f'C{i}', label=f'C{i + 1}')
#
#     # 添加行星节点（中层）
#     dot.attr('node', fillcolor='lightgreen')
#     for i in range(n_p):
#         dot.node(f'P{i}', label=f'P{i + 1}')
#
#     # 添加太阳轮节点（底层）
#     dot.attr('node', fillcolor='lightcoral')
#     for i in range(n_s):
#         dot.node(f'S{i}', label=f'S{i + 1}')
#
#     # 添加层级分隔 - 修复层级分离问题
#     # 创建三个独立的层级组
#     with dot.subgraph(name='cluster_carriers') as s:
#         s.attr(rank='max')
#         s.attr(style='invis')  # 隐藏子图边框
#         for i in range(n_c):
#             s.node(f'C{i}')
#
#     with dot.subgraph(name='cluster_planets') as s:
#         s.attr(rank='same')
#         s.attr(style='invis')  # 隐藏子图边框
#         for i in range(n_p):
#             s.node(f'P{i}')
#
#     with dot.subgraph(name='cluster_suns') as s:
#         s.attr(rank='min')
#         s.attr(style='invis')  # 隐藏子图边框
#         for i in range(n_s):
#             s.node(f'S{i}')
#
#     # 添加层级之间的不可见边，确保正确的纵向排序
#     if n_c > 0 and n_p > 0:
#         dot.edge(f'C0', f'P0', style='invis')
#     if n_p > 0 and n_s > 0:
#         dot.edge(f'P0', f'S0', style='invis')
#     elif n_c > 0 and n_s > 0:
#         dot.edge(f'C0', f'S0', style='invis')
#
#     # 添加转动副连接（实线）
#     planet_offset = 0
#     for carrier_idx, planet_count in enumerate(graph["turning_partition"]):
#         for j in range(planet_count):
#             planet_idx = planet_offset + j
#             dot.edge(f'C{carrier_idx}', f'P{planet_idx}',
#                      color='black', style='solid', penwidth='2.0')
#         planet_offset += planet_count
#
#     # 添加载体-行星齿轮副（虚线）
#     for carrier_idx, planet_idx in graph["gear_edges"]:
#         dot.edge(f'C{carrier_idx}', f'P{planet_idx}',
#                  color='red', style='dashed', penwidth='2.0')
#
#     # 添加太阳轮-行星齿轮副（虚线）
#     for sun_idx, planet_idx in graph["sun_edges"]:
#         dot.edge(f'S{sun_idx}', f'P{planet_idx}',
#                  color='blue', style='dashed', penwidth='2.0')
#
#     return dot

def create_graphviz_diagram(graph):
    dot = graphviz.Digraph(
        format='png',
        graph_attr={'rankdir':'TB', 'nodesep':'0.5','ranksep':'1'},
        node_attr={'shape':'circle','style':'filled','fixedsize':'true','width':'0.6','height':'0.6'},
        edge_attr={'arrowhead':'none'}
    )

    # 载体 C 层
    dot.attr('node', fillcolor='lightblue')
    with dot.subgraph() as s:
        s.attr(rank='source')
        for i in range(graph['n_c']):
            s.node(f'C{i}', label=f'C{i+1}')

    # 行星 P 层
    dot.attr('node', fillcolor='lightgreen')
    with dot.subgraph() as s:
        s.attr(rank='same')
        for i in range(graph['n_p']):
            s.node(f'P{i}', label=f'P{i+1}')

    # 太阳轮 S 层
    dot.attr('node', fillcolor='lightcoral')
    with dot.subgraph() as s:
        s.attr(rank='sink')
        for i in range(graph['n_s']):
            s.node(f'S{i}', label=f'S{i+1}')

    # 隐形边，强制 C→P→S 分层
    if graph['n_c'] and graph['n_p']:
        dot.edge('C0', 'P0', style='invis')
    if graph['n_p'] and graph['n_s']:
        dot.edge('P0', 'S0', style='invis')

    # …（接着加你的实线、虚线边）…
    # 转动副
    planet_offset = 0
    for ci, cnt in enumerate(graph['turning_partition']):
        for j in range(cnt):
            pi = planet_offset + j
            dot.edge(f'C{ci}', f'P{pi}', color='black', style='solid', penwidth='2')
        planet_offset += cnt
    # 载体–行星齿轮副
    for ci, pi in graph['gear_edges']:
        dot.edge(f'C{ci}', f'P{pi}', color='red', style='dashed', penwidth='2')
    # 太阳轮–行星齿轮副
    for si, pi in graph['sun_edges']:
        dot.edge(f'S{si}', f'P{pi}', color='blue', style='dashed', penwidth='2')

    return dot


def save_graphs_to_csv(graphs, filename="pgt_complete_graphs.csv"):
    """将完整图保存到CSV文件"""
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "n", "n_c", "n_p", "n_s", "turning_partition",
            "gear_edges", "sun_edges", "sun_assignment", "planet_assignment",
            "det_signature", "image_path"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n, graph_list in graphs.items():
            for idx, graph in enumerate(graph_list):
                # 生成图像文件名
                img_filename = f"pgt_n{n}_g{idx + 1}.png"
                img_path = os.path.join("pgt_graphs", img_filename)

                row = {
                    "n": n,
                    "n_c": graph["n_c"],
                    "n_p": graph["n_p"],
                    "n_s": graph["n_s"],
                    "turning_partition": str(graph["turning_partition"]),
                    "gear_edges": str(graph["gear_edges"]),
                    "sun_edges": str(graph["sun_edges"]),
                    "sun_assignment": str(graph["sun_assignment"]),
                    "planet_assignment": str(graph["planet_assignment"]),
                    "det_signature": f"{graph['det_signature']:.12f}",
                    "image_path": img_path
                }
                writer.writerow(row)


def save_and_display_graphs(graphs, directory="pgt_graphs", format='png'):
    """
    将完整图保存为图像文件并在PyCharm中显示

    参数:
        graphs: 生成的图字典
        directory: 保存图像的目录
        format: 图像格式 ('png', 'jpg', 'svg', 'pdf'等)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    saved_images = []

    for n, graph_list in graphs.items():
        for idx, graph in enumerate(graph_list):
            try:
                # 创建graphviz图
                dot = create_graphviz_diagram(graph)

                # 设置文件名
                filename = f"pgt_n{n}_g{idx + 1}"
                filepath = os.path.join(directory, filename)

                # 渲染图像
                img_path = dot.render(filepath, format=format, cleanup=True)
                saved_images.append(img_path)
                print(f"已保存: {img_path}")
            except graphviz.backend.execute.ExecutableNotFound as e:
                print(f"错误: {e}")
                print("请确保已安装Graphviz软件并将其添加到系统PATH")
                print("下载地址: https://graphviz.org/download/")
                return saved_images
            except Exception as e:
                print(f"生成图像时出错: {e}")
                continue

    return saved_images


def open_image(image_path):
    """根据操作系统打开图像文件"""
    try:
        if platform.system() == 'Windows':
            os.startfile(image_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', image_path))
        else:  # Linux
            subprocess.call(('xdg-open', image_path))
        return True
    except Exception as e:
        print(f"无法打开图像: {e}")
        return False


def main():
    """主函数：生成、保存并显示PGT图"""
    print("行星齿轮系完整图生成系统 (PyCharm版)")
    print("基于: Enumeration of 1-DOF Planetary Gear Train Graphs Based on Functional Constraints")
    print("=" * 80)
    print("布局说明:")
    print("- 三层纵向布局: 载体(C) → 行星(P) → 太阳轮(S)")
    print("- 所有构件用圆圈表示")
    print("- 转动副: 黑色实线 | 齿轮副: 红色/蓝色虚线")
    print("- 移除多重转动副(MR)表示")
    print("- 所有边无箭头")
    print("=" * 80)

    # 检查依赖
    try:
        import numpy
    except ImportError:
        print("错误: 缺少必要依赖 - numpy")
        print("请安装: pip install numpy")
        return

    # 检查Graphviz安装
    print("\n检查Graphviz安装...")
    if not check_graphviz_installation():
        print("\n[错误] Graphviz 未安装或未正确配置!")
        print("请执行以下步骤:")
        print("1. 下载并安装 Graphviz 软件: https://graphviz.org/download/")
        print("2. 安装时勾选 'Add Graphviz to the system PATH for current user'")
        print("3. 如果已安装，请手动将Graphviz的bin目录添加到系统PATH")
        print("   例如: C:\\Program Files\\Graphviz\\bin")
        print("4. 重启PyCharm后再次运行此程序")
        return

    # 生成n=4到10的完整图
    print("\n[1/3] 生成行星齿轮系图...")
    complete_graphs = generate_complete_graphs(11, 13)

    total_count = sum(len(graphs) for graphs in complete_graphs.values())
    print(f"\n总共生成 {total_count} 个完整图")

    # 按构件数统计
    for n in sorted(complete_graphs.keys()):
        graphs = complete_graphs[n]
        print(f"n={n}: {len(graphs)} 个图")

    # 保存到CSV
    print("\n[2/3] 保存结果到CSV文件...")
    save_graphs_to_csv(complete_graphs, "pgt_complete_graphs.csv")
    print("结果已保存到 pgt_complete_graphs.csv")

    # 保存为图像文件
    print("\n[3/3] 生成并显示图像...")
    saved_images = save_and_display_graphs(complete_graphs, "pgt_graphs", 'png')

    if not saved_images:
        print("未生成任何图像，程序终止")
        return

    # 尝试打开第一个图像
    print("\n尝试在默认图像查看器中打开示例图像...")
    if open_image(saved_images[0]):
        print(f"已打开: {saved_images[0]}")
    else:
        print("无法自动打开图像，请手动查看以下文件:")
        print(saved_images[0])

    # 提供所有图像路径
    print("\n所有生成的图像路径:")
    for img in saved_images:
        print(f"- {img}")

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
