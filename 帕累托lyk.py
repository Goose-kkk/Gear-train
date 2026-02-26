import matplotlib
matplotlib.use('TkAgg')

import os
import csv

import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ========== 修复中文显示乱码问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 优先使用无衬线字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 全局模数标准系列 ==========
# 对应索引: 0    1     2    3    4    5    6    7    8    9   10
M_STANDARD = [1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# ========== 全局配置：齿形系数与应力修正系数表 (表10-5) ==========
# 格式: {齿数 z: (Y_Fa, Y_Sa)}
GEAR_FACTOR_TABLE = {
    17: (2.97, 1.52),
    18: (2.91, 1.53),
    19: (2.85, 1.54),
    20: (2.80, 1.55),
    21: (2.76, 1.56),
    22: (2.72, 1.57),
    23: (2.69, 1.575),
    24: (2.65, 1.58),
    25: (2.62, 1.59),
    26: (2.60, 1.595),
    27: (2.57, 1.60),
    28: (2.55, 1.61),
    29: (2.53, 1.62),
    30: (2.52, 1.625),
    35: (2.45, 1.65),
    40: (2.40, 1.67),
    45: (2.35, 1.68),
    50: (2.32, 1.70),
    60: (2.28, 1.73),
    70: (2.24, 1.75),
    80: (2.22, 1.77),
    90: (2.20, 1.78),
    100: (2.18, 1.79),
    150: (2.14, 1.83),
    200: (2.12, 1.865),
    # 200以上按200处理，或可视情况增加无穷大项(2.06, 1.97)
}


def get_geometry_factors(z):
    """
    根据齿数 z 查表获取 (Y_Fa, Y_Sa)。
    如果 z 在表格区间内，采用线性插值计算。
    """
    z = int(z)

    # 1. 边界处理
    sorted_keys = sorted(GEAR_FACTOR_TABLE.keys())
    if z <= sorted_keys[0]:
        return GEAR_FACTOR_TABLE[sorted_keys[0]]
    if z >= sorted_keys[-1]:
        return GEAR_FACTOR_TABLE[sorted_keys[-1]]

    # 2. 精确匹配
    if z in GEAR_FACTOR_TABLE:
        return GEAR_FACTOR_TABLE[z]

    # 3. 线性插值
    # 找到 z 左右两边的 key
    idx = 0
    for i, key in enumerate(sorted_keys):
        if key > z:
            idx = i
            break

    x1 = sorted_keys[idx - 1]
    x2 = sorted_keys[idx]
    y_fa1, y_sa1 = GEAR_FACTOR_TABLE[x1]
    y_fa2, y_sa2 = GEAR_FACTOR_TABLE[x2]

    # 插值公式: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    factor = (z - x1) / (x2 - x1)
    y_fa = y_fa1 + factor * (y_fa2 - y_fa1)
    y_sa = y_sa1 + factor * (y_sa2 - y_sa1)

    return y_fa, y_sa

def fitness0(individual):
    """
    计算行星齿轮系统的两个目标函数值
    输入 individual 包含 8 个变量：[Z1, Z2, Z3, Z5, Z6, phi_d1, phi_d2,m_index]
    """
    # ================= 提取设计变量（Z为整数, phi_d为浮点） =================
    # 提取 Z 变量 (整数)
    Z1, Z2, Z3, Z5, Z6 = individual[:5].astype(int)
    # 提取 phi_d 变量 (浮点数)
    phi_d1, phi_d2 = individual[5:7]  # 索引 5 和 6
    # m (从列表中取值)
    m_idx = int(round(individual[7]))
    # 防止越界保护
    m_idx = max(0, min(m_idx, len(M_STANDARD)-1))
    m = M_STANDARD[m_idx]

    #示例值需要修改
    # m = 2  # 模数（整数）
    rho = 7800  # 材料密度 (kg/m³)
    g = 9.81  # 重力加速度 (m/s²)
    fm = 0.06  # 摩擦系数
    T1 = 10000.0  # 输入转矩 T1 (N·mm) <--- 请修改这里,
    K_F = 1.2  # 动载系数 KF <--- 请修改这里
    w = 10  # 转速 (rpm) [示例值]
    L_h = 20000.0  # 设计寿命 (小时) [示例值]
    sigma_flim = 335  # 材料弯曲疲劳极限应力 (MPa)
    # sigma_hlim=390
    S=1#安全系数
    j_n=1
    # Y_epsilon1 = 0.25+0.75/ e51 # 重合度系数
    # Y_epsilon2 = 0.25+0.75/ e63


    # ================= 内部计算 b 变量 (按公式) =================
    # b1, b3, b5, b6, b2 现在是计算结果，是浮点数

    # 1. b1 = b5 = min{Z1, Z5} * m * phi_d1
    b5 = min(Z1, Z5) * m * phi_d1
    b1 = b5

    # 2. b3 = b6 = min{Z3, Z6} * m * phi_d2
    b6 = min(Z3, Z6) * m * phi_d2
    b3 = b6

    b2 = b5 + b6

    # 关键约束检查（Z1, Z3 的小范围调整）
    # 这一调整是为了防止效率计算中分母为零，但应当确保 Z1, Z2, Z3, Z5, Z6 满足齿数约束。
    # NSGA2.generate_valid_gear_teeth 已经确保了齿数约束。
    # 此处只保留 Z1 == Z3 的非零调整，防止计算崩溃
    if Z1 == Z3:
        # Z1必须是整数，如果Z1=Z3，会导致效率公式的分母为零（j=Z1/(Z1-Z3)）
        # 如果 Z1, Z3 是变量，这个调整是必要的。
        Z1 = Z1 + 1 if Z1 < 500 else Z1 - 1

    # --- 移除 'if b2 <= b5 + b6:' 块，因为它没有意义且导致了 UnboundLocalError ---

    # ================= 内部辅助函数 (保持不变) =================
    def calculate_alpha_6(z_6):
        # ... (保持不变) ...
        cos_val = (z_6 / (z_6 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_5(z_5):
        # ... (保持不变) ...
        cos_val = (-z_5 / (z_5 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_1(z_1):
        # ... (保持不变) ...
        cos_val = (z_1 / (z_1 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_2(z_2):
        # ... (保持不变) ...
        cos_val = (-z_2 / (z_2 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_3(z_3):
        # ... (保持不变) ...
        cos_val = (z_3 / (z_3 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    # ... (效率计算部分 f1 保持不变) ...
    alpha = math.radians(20)
    a1 = calculate_alpha_1(Z1)
    a2 = calculate_alpha_2(Z2)
    a3 = calculate_alpha_3(Z3)
    a5 = calculate_alpha_5(Z5)
    a6 = calculate_alpha_6(Z6)

    # ... (计算epsilon, lambda, k, j 保持不变) ...
    e62 = (Z6 * (math.tan(a6) - math.tan(alpha)) - Z2 * (math.tan(alpha) - math.tan(a2))) / (2 * math.pi)
    e51 = (Z5 * (math.tan(a5) - math.tan(alpha)) + Z1 * (math.tan(a1) - math.tan(alpha))) / (2 * math.pi)
    e63 = (Z6 * (math.tan(a6) - math.tan(alpha)) + Z3 * (math.tan(a3) - math.tan(alpha))) / (2 * math.pi)
    e52 = (Z5 * (math.tan(a5) - math.tan(alpha)) - Z2 * (math.tan(a2) - math.tan(alpha))) / (2 * math.pi)

    # 现在计算 Y_epsilon (重合度系数)
    Y_epsilon1 = 0.25 + 0.75 / e51 if e51 > 0 else 1.0
    Y_epsilon2 = 0.25 + 0.75 / e63 if e63 > 0 else 1.0

    # ================= 动态查表获取系数 =================
    # 针对太阳轮 Z1 进行查表 (公式中用的是 Z1)
    Y_Fa, Y_Sa = get_geometry_factors(Z1)
    # ================= 3. 模数强度校验  =================

    # (1) 计算应力循环次数 N = 60 * w * j * L_h
    N_cycles = 60 * w * j_n * L_h

    # (2) 计算寿命系数 K_N
    # 引用图片公式: K_N = 3.2 - 0.41 * log10(N)
    # 用户约束: N 范围 (100, 1000000)。若超出此范围，公式可能失效，这里做截断处理。
    N_calc = np.clip(N_cycles, 100, 1000000)
    K_N = 3.2 - 0.41 * np.log10(N_calc)

    # (3) 计算许用弯曲应力 [sigma_F]
    # 引用图片公式: [sigma_F] = (K_N * sigma_lim) / S
    sigma_F_allow = (K_N * sigma_flim) / S# 许用弯曲应力 [sigma_F] (MPa)

    # ================= 2. 模数约束检查 =================
    # 公式：m >= ( (2 * K_F * T1 * Y_eps) / (phi_d * Z1^2) * (Y_Fa * Y_Sa / sigma_F) ) ^ (1/3)
    # 注意：防止除零错误
    if phi_d1 <= 0 or Z1 <= 0 or sigma_F_allow  <= 0:
        return 1e9, 1e9  # 返回极大惩罚值

    term1 = (2 * K_F * T1 * Y_epsilon1) / (phi_d1 * (Z1 ** 2))
    term3= (2 * K_F * T1 * Y_epsilon2) / (phi_d2 * (Z1 ** 2))
    term2 = (Y_Fa * Y_Sa) / sigma_F_allow
    min_m_val_1 = (term1 * term2) ** (1 / 3.0)
    min_m_val_2 = (term1 * term3) ** (1 / 3.0)
    # 如果选取的 m 小于计算出的最小 m，则给予惩罚（使其被淘汰）
    if m <max(min_m_val_1,  min_m_val_2 ) :
        # 这是一个违反约束的解，返回极大的目标值
        return 1e9, 1e9


    lambda1 = (math.pi / 2) * e62 * fm * (1 / Z6 - 1 / Z2) if Z6 != Z2 else 0.0
    lambda2 = (math.pi / 2) * e52 * fm * (1 / Z5 - 1 / Z2) if Z5 != Z2 else 0.0
    lambda3 = (math.pi / 2) * e63 * fm * (1 / Z6 + 1 / Z3)
    lambda4 = (math.pi / 2) * e51 * fm * (1 / Z5 + 1 / Z1)

    k_numerator = Z1 * (Z2 + Z3)
    k_denominator = Z2 * (Z1 - Z3)
    k = k_numerator / k_denominator if k_denominator != 0 else 1.0

    j_denominator = Z1 - Z3
    j = Z1 / j_denominator if j_denominator != 0 else 2.0

    # 最终效率计算
    num_part1 = (1 - lambda1) * lambda2 + lambda1
    num_part2 = num_part1 * (1 - lambda4)
    num_part3 = lambda3 * (1 - lambda1) * (1 - lambda2 * (1 - lambda4) - lambda4)
    num_total = (num_part2 + num_part3 + lambda4) * (1 - (lambda3 * (1 - lambda1) + lambda1) * (1 - j / k))

    den_part1 = 1 / (j - 1) if j != 1 else 0.0
    den_part2 = lambda4 + lambda2 * (1 - lambda4)
    den_part3 = (lambda3 * (1 - lambda1) + lambda1) * (1 - lambda2 * (1 - lambda4) - lambda4)
    den_total = den_part1 + den_part2 + den_part3

    # 计算效率
    if den_total != 0:
        efficiency = 1 - abs(num_total / den_total + (lambda1 + lambda3 * (1 - lambda1) + lambda1) * (1 - j / k))
    else:
        efficiency = 0.7

        # 限制效率范围
    efficiency = np.clip(efficiency, 0, 1)
    f1 = 1.0 - efficiency  # 目标1：1-效率（越小越好）

    # ================= 重量计算 (f2) =================
    # 计算各部分直径（基于整数Z和m）
    d1 = m * Z1
    d3 = m * Z3
    d5 = m * Z5
    d6 = m * Z6
    d2a = m * (Z2 - 2)
    d2f = m * (Z2 + 2 * 1.25)

    # 计算各部分体积（基于浮点数b1, b2, b3, b5, b6）
    V_sun = (math.pi / 4) * (b1 * (d1 ** 2) + b3 * (d3 ** 2))
    V_ring = (math.pi / 4) * b2 * (d2f ** 2 - d2a ** 2)
    V_planet = (math.pi / 4) * 3 * (b5 * (d5 ** 2) + b6 * (d6 ** 2))

    # 总体积（mm³ → m³）
    total_volume = (V_sun + V_ring + V_planet) * 1e-9
    # 重量计算（kg → N）
    mass = total_volume * rho
    f2 = mass * g  # 目标2：重量（越小越好）

    return f1, f2

class NSGA2:
    def __init__(self,
                 fitness_func,
                 n_var=8,  # 变量总数改为 8
                 n_obj=2,
                 pop_size=30,
                 n_gen=100,
                 crossover_prob=0.9,
                 mutation_prob=0.3,
                 eta_c=20,
                 eta_m=20,
                 archive_size=50,
                 lower_bounds=None,
                 upper_bounds=None,
                 initial_archive=None):

        self.fitness_func = fitness_func
        self.n_var = n_var
        self.n_obj = n_obj
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.archive_size = archive_size

        # 变量上下界
        if lower_bounds is None:
            # 5个Z + 2个phi + 1个m_index
            self.lower_bounds = np.array([18, 18, 18, 18, 18, 0.2, 0.2, 0], dtype=np.float64)
        else:
            self.lower_bounds = np.array(lower_bounds, dtype=np.float64)

        if upper_bounds is None:
            # m_index 的上限是标准数组长度减 1
            self.upper_bounds = np.array([100, 150, 100, 50, 40, 1.4, 1.4, len(M_STANDARD) - 1], dtype=np.float64)
        else:
            self.upper_bounds = np.array(upper_bounds, dtype=np.float64)

        self.archive = []
        self.archive_fitness = []

        # 初始化外部归档集（如果提供了初始解）
        if initial_archive is not None:
            # 确保初始归档集与新的 7 维变量匹配
            initial_archive = [ind[:self.n_var] for ind in initial_archive if len(ind) >= self.n_var]
            self.initialize_external_archive(initial_archive)

    def dominates(self, fitness_a, fitness_b):
        """判断a是否支配b"""
        all_less_or_equal = all(fitness_a[i] <= fitness_b[i] for i in range(self.n_obj))
        at_least_one_less = any(fitness_a[i] < fitness_b[i] for i in range(self.n_obj))
        return all_less_or_equal and at_least_one_less

    def initialize_external_archive(self, initial_archive):
        """
        初始化外部归档集
        参数:
            initial_archive: 初始解列表或数组，每个解是一个7维向量
        """
        if len(initial_archive) == 0:
            return

        # 确保初始归档集是numpy数组格式
        initial_archive = np.array(initial_archive, dtype=np.float64)

        # 计算所有初始解的适应度
        initial_fitness = []
        for ind in initial_archive:
            try:
                f1, f2 = self.fitness_func(ind)
                initial_fitness.append([f1, f2])
            except Exception as e:
                # print(f"警告：计算初始解适应度时出错: {e}")
                continue

        if len(initial_fitness) == 0:
            return

        initial_fitness = np.array(initial_fitness, dtype=np.float64)

        # 1. 非支配排序，找出所有非支配解（第一前沿）
        fronts, _ = self.fast_non_dominated_sort(initial_fitness)

        if len(fronts) == 0:
            return

        first_front = fronts[0]

        # 2. 将所有第一前沿解加入归档集
        for idx in first_front:
            if np.all(np.isfinite(initial_fitness[idx])):
                is_dominated = False
                for arch_fit in self.archive_fitness:
                    if self.dominates(arch_fit, initial_fitness[idx]):
                        is_dominated = True
                        break

                if not is_dominated:
                    to_remove = []
                    for i, arch_fit in enumerate(self.archive_fitness):
                        if self.dominates(initial_fitness[idx], arch_fit):
                            to_remove.append(i)

                    for i in sorted(to_remove, reverse=True):
                        del self.archive[i]
                        del self.archive_fitness[i]

                    is_duplicate = False
                    for arch_ind in self.archive:
                        # 使用 np.allclose 检查浮点数组是否近似相等
                        if np.allclose(arch_ind, initial_archive[idx], atol=1e-4):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        self.archive.append(initial_archive[idx])
                        self.archive_fitness.append(initial_fitness[idx])

        # 3. 如果归档集超过大小限制，按拥挤距离裁剪
        if len(self.archive) > self.archive_size:
            self.trim_archive()

    def add_to_archive(self, individual, fitness):
        """单个个体加入归档集"""
        is_dominated = False
        for arch_fit in self.archive_fitness:
            if self.dominates(arch_fit, fitness):
                is_dominated = True
                break

        if is_dominated:
            return False

        to_remove = []
        for idx, arch_fit in enumerate(self.archive_fitness):
            if self.dominates(fitness, arch_fit):
                to_remove.append(idx)

        for idx in sorted(to_remove, reverse=True):
            del self.archive[idx]
            del self.archive_fitness[idx]

        is_duplicate = False
        for arch_ind in self.archive:
            if np.allclose(arch_ind, individual, atol=1e-4):
                is_duplicate = True
                break

        if not is_duplicate:
            self.archive.append(individual)
            self.archive_fitness.append(fitness)

        if len(self.archive) > self.archive_size:
            self.trim_archive()

        return True

    def trim_archive(self):
        """按拥挤距离裁剪归档集"""
        if len(self.archive) <= self.archive_size:
            return

        fitness_np = np.array(self.archive_fitness)
        distances = self.calculate_crowding_distance(fitness_np)

        sorted_indices = np.argsort(-distances)
        keep_indices = sorted_indices[:self.archive_size]

        self.archive = [self.archive[i] for i in keep_indices]
        self.archive_fitness = [self.archive_fitness[i] for i in keep_indices]

    def calculate_crowding_distance(self, fitness_matrix):
        """计算拥挤距离"""
        n = len(fitness_matrix)
        distances = np.zeros(n)

        if n <= 2:
            return np.ones(n) * float('inf')

        for obj_idx in range(self.n_obj):
            sorted_indices = np.argsort(fitness_matrix[:, obj_idx])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            f_min = fitness_matrix[sorted_indices[0], obj_idx]
            f_max = fitness_matrix[sorted_indices[-1], obj_idx]

            if f_max - f_min < 1e-10:
                continue

            for i in range(1, n - 1):
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                distances[sorted_indices[i]] += (fitness_matrix[next_idx, obj_idx] -
                                                 fitness_matrix[prev_idx, obj_idx]) / (f_max - f_min)

        return distances

    def remove_duplicates(self, population):
        """移除种群中的重复个体，适应浮点和整数混合的个体"""
        unique_tuples = []
        unique_indices = []

        # 四舍五入到一定精度进行比较，以应对浮点数重复问题
        rounded_pop = np.round(population, decimals=4)

        for idx, ind in enumerate(rounded_pop):
            ind_tuple = tuple(ind)
            if ind_tuple not in unique_tuples:
                unique_tuples.append(ind_tuple)
                unique_indices.append(idx)

        unique_pop = population[unique_indices]
        if len(unique_pop) < len(population):
            need_ = len(population) - len(unique_pop)
            new_individuals = [self.create_valid_individual() for _ in range(need_)]
            new_individuals = np.array(new_individuals, dtype=np.float64)
            unique_pop = np.vstack([unique_pop, new_individuals])

        return unique_pop

    def check_gear_constraints(self, Z1, Z2, Z3, Z5, Z6):
        """检查齿轮齿数约束是否满足"""
        # 核心齿数约束
        constraint1 = (Z1 + 2 * Z5) == Z2  # Z1+2*Z5 == Z2
        constraint2 = (Z3 + 2 * Z6) == Z2  # Z3+2*Z6 == Z2
        constraint3 = (Z1 + Z5) > (Z3 + Z6)  # Z1+Z5 > Z3+Z6
        constraint4 = Z1 > Z3  # Z1 > Z3
        constraint5 = Z5 < Z6  # Z5 < Z6

        # 边界约束 (索引 0-4)
        constraint6 = (self.lower_bounds[0] <= Z1 <= self.upper_bounds[0])
        constraint7 = (self.lower_bounds[1] <= Z2 <= self.upper_bounds[1])
        constraint8 = (self.lower_bounds[2] <= Z3 <= self.upper_bounds[2])
        constraint9 = (self.lower_bounds[3] <= Z5 <= self.upper_bounds[3])
        constraint10 = (self.lower_bounds[4] <= Z6 <= self.upper_bounds[4])

        return (constraint1 and constraint2 and constraint3 and
                constraint4 and constraint5 and constraint6 and
                constraint7 and constraint8 and constraint9 and constraint10)

    def generate_valid_gear_teeth(self):
        """生成满足所有齿数约束的齿轮参数（返回整数）"""
        max_attempts = 1000
        for _ in range(max_attempts):
            # 1. 先确定Z5和Z6，确保Z5 < Z6且在边界内
            z5_max = min(int(self.upper_bounds[3]), int(self.upper_bounds[4]) - 1)
            if z5_max <= int(self.lower_bounds[3]):
                z5_max = int(self.lower_bounds[3])

            Z5 = np.random.randint(int(self.lower_bounds[3]), z5_max + 1)
            Z6 = np.random.randint(Z5 + 1, int(self.upper_bounds[4]) + 1)

            # 2. 确定Z3和Z1，确保Z1 > Z3且在边界内
            z3_max = min(int(self.upper_bounds[2]), int(self.upper_bounds[0]) - 1)
            if z3_max <= int(self.lower_bounds[2]):
                z3_max = int(self.lower_bounds[2])

            Z3 = np.random.randint(int(self.lower_bounds[2]), z3_max + 1)
            Z1 = np.random.randint(Z3 + 1, int(self.upper_bounds[0]) + 1)

            # 3. 根据约束计算Z2，并检查Z2边界
            Z2_from_Z1 = Z1 + 2 * Z5
            Z2_from_Z3 = Z3 + 2 * Z6

            if Z2_from_Z1 == Z2_from_Z3:
                Z2 = Z2_from_Z1
                if (int(self.lower_bounds[1]) <= Z2 <= int(self.upper_bounds[1]) and
                        (Z1 + Z5) > (Z3 + Z6)):
                    if self.check_gear_constraints(Z1, Z2, Z3, Z5, Z6):
                        return Z1, Z2, Z3, Z5, Z6

        # 返回安全的默认值
        return 20, 60, 18, 20, 21

    def create_valid_individual(self):
        """创建有效个体（5个Z整数, 2个phi_d浮点数）"""
        max_attempts = 100
        for _ in range(max_attempts):
            # 1. 生成满足齿数约束的整数参数 (Z1, Z2, Z3, Z5, Z6)
            Z1, Z2, Z3, Z5, Z6 = self.generate_valid_gear_teeth()

            # 2. 生成浮点参数 (phi_d1, phi_d2) (索引 5, 6)
            phi_d1 = np.random.uniform(self.lower_bounds[5], self.upper_bounds[5])
            phi_d2 = np.random.uniform(self.lower_bounds[6], self.upper_bounds[6])

            # 3. 生成模数索引 m_index (对应 M_STANDARD 的长度)
            m_idx = np.random.randint(0, len(M_STANDARD))

            # 构建完整个体，确保包含 8 个变量
            individual = np.array([Z1, Z2, Z3, Z5, Z6, phi_d1, phi_d2, m_idx], dtype=np.float64)

            # 验证个体有效性
            try:
                f1, f2 = self.fitness_func(individual)
                if np.isfinite(f1) and np.isfinite(f2):
                    return individual
            except Exception:
                continue
                # 默认个体也要补齐到 8 位
            return np.array([20, 60, 18, 20, 21, 1.0, 1.0, 2], dtype=np.float64)

    def initialize_population(self):
        """初始化种群"""
        population = []
        for i in range(self.pop_size):
            population.append(self.create_valid_individual())
        population = np.array(population, dtype=np.float64)

        population = self.remove_duplicates(population)
        fitness = self.evaluate_population(population)

        fronts, _ = self.fast_non_dominated_sort(fitness)
        first_front = fronts[0] if fronts else []

        for idx in first_front:
            self.add_to_archive(population[idx], fitness[idx])

        return population

    def evaluate_population(self, population):
        """评估种群适应度"""
        fitness = []
        for ind in population:
            f1, f2 = self.fitness_func(ind)
            fitness.append([f1, f2])
        return np.array(fitness, dtype=np.float64)

    def fast_non_dominated_sort(self, fitness):
        """快速非支配排序"""
        n = len(fitness)
        S = [[] for _ in range(n)]
        n_p = [0] * n
        rank = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(fitness[i], fitness[j]):
                        S[i].append(j)
                    elif self.dominates(fitness[j], fitness[i]):
                        n_p[i] += 1

            if n_p[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_p[q] -= 1
                    if n_p[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break

        return fronts, rank

    def selection(self, population, fitness, fronts):
        """锦标赛选择"""
        selected = []
        while len(selected) < self.pop_size:
            candidates = np.random.choice(len(population), 3, replace=False)

            best_idx = candidates[0]
            best_rank = next(r for r, front in enumerate(fronts) if best_idx in front)

            for idx in candidates[1:]:
                current_rank = next(r for r, front in enumerate(fronts) if idx in front)
                if current_rank < best_rank:
                    best_idx = idx
                    best_rank = current_rank
                elif current_rank == best_rank:
                    front_indices = fronts[best_rank]
                    fitness_subset = fitness[front_indices]
                    distances = self.calculate_crowding_distance(fitness_subset)

                    pos1 = front_indices.index(best_idx) if best_idx in front_indices else -1
                    pos2 = front_indices.index(idx) if idx in front_indices else -1

                    if pos1 >= 0 and pos2 >= 0 and distances[pos2] > distances[pos1]:
                        best_idx = idx

            selected.append(best_idx)

        return population[selected]

    def crossover(self, parent1, parent2):
        """混合交叉操作 (整数 Z + 浮点 phi_d)"""
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        # 交叉点选择 (索引 0 到 6)
        cross_points = np.random.choice(range(self.n_var), size=2, replace=False)
        for point in cross_points:
            child1[point], child2[point] = child2[point], child1[point]

        # 修复约束和类型
        child1 = self.repair_individual(child1)
        child2 = self.repair_individual(child2)

        # 验证并修复齿数约束
        for child in [child1, child2]:
            Z1, Z2, Z3, Z5, Z6 = [int(z) for z in child[:5]]
            if not self.check_gear_constraints(Z1, Z2, Z3, Z5, Z6):
                new_Z1, new_Z2, new_Z3, new_Z5, new_Z6 = self.generate_valid_gear_teeth()
                child[:5] = [float(new_Z1), float(new_Z2), float(new_Z3), float(new_Z5), float(new_Z6)]

        return child1, child2

    def mutation(self, individual):
        """混合变异操作 (整数 Z + 浮点 phi_d)"""
        mutated = individual.copy()

        for i in range(self.n_var):  # n_var = 7
            if np.random.random() < self.mutation_prob:
                if i < 5:  # 齿数参数 (Z1, Z2, Z3, Z5, Z6)
                    # 重新生成整个齿数组合以保持约束
                    Z1, Z2, Z3, Z5, Z6 = self.generate_valid_gear_teeth()
                    mutated[:5] = [float(Z1), float(Z2), float(Z3), float(Z5), float(Z6)]
                    break
                else:  # 浮点参数 (phi_d1, phi_d2)
                    mutation_step = np.random.uniform(-0.1, 0.1)
                    new_val = mutated[i] + mutation_step
                    mutated[i] = np.clip(new_val, self.lower_bounds[i], self.upper_bounds[i])

        # 修复约束和类型
        mutated = self.repair_individual(mutated)

        # 最终验证齿数约束
        Z1, Z2, Z3, Z5, Z6 = [int(z) for z in mutated[:5]]
        if not self.check_gear_constraints(Z1, Z2, Z3, Z5, Z6):
            new_Z1, new_Z2, new_Z3, new_Z5, new_Z6 = self.generate_valid_gear_teeth()
            mutated[:5] = [float(new_Z1), float(new_Z2), float(new_Z3), float(new_Z5), float(new_Z6)]

        return mutated

    def repair_individual(self, individual):
        """修复个体约束并强制整数类型"""
        # 1. 强制 Z 变量为整数 (索引 0-4)
        individual[:5] = np.round(individual[:5]).astype(np.float64)

        # 2. 修复所有变量的边界约束
        for i in range(self.n_var):
            individual[i] = np.clip(individual[i], self.lower_bounds[i], self.upper_bounds[i])

        # 3. 再次强制 Z 变量为整数 (确保 float 类型存储的是整数值)
        individual[:5] = np.round(individual[:5]).astype(np.float64)

        return individual

    def evolve(self):
        """主进化过程"""
        population = self.initialize_population()
        fitness = self.evaluate_population(population)

        for gen in range(self.n_gen):
            fronts, _ = self.fast_non_dominated_sort(fitness)
            parents = self.selection(population, fitness, fronts)

            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    p1 = parents[i]
                    p2 = parents[i + 1]
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.mutation(c1)
                    c2 = self.mutation(c2)
                    offspring.extend([c1, c2])

            while len(offspring) < self.pop_size:
                offspring.append(self.create_valid_individual())

            offspring = np.array(offspring[:self.pop_size], dtype=np.float64)
            offspring = self.remove_duplicates(offspring)
            offspring_fitness = self.evaluate_population(offspring)

            offspring_fronts, _ = self.fast_non_dominated_sort(offspring_fitness)
            offspring_first_front = offspring_fronts[0] if offspring_fronts else []
            for idx in offspring_first_front:
                self.add_to_archive(offspring[idx], offspring_fitness[idx])

            combined_pop = np.vstack([population, offspring])
            combined_pop = self.remove_duplicates(combined_pop)
            combined_fit = self.evaluate_population(combined_pop)

            combined_fronts, _ = self.fast_non_dominated_sort(combined_fit)
            new_pop = []
            new_fit = []
            front_idx = 0

            while len(new_pop) < self.pop_size and front_idx < len(combined_fronts):
                current_front = combined_fronts[front_idx]
                if len(current_front) == 0:
                    front_idx += 1
                    continue

                front_fitness = combined_fit[current_front]
                distances = self.calculate_crowding_distance(front_fitness)
                sorted_indices = np.argsort(-distances)

                for idx in sorted_indices:
                    if len(new_pop) >= self.pop_size:
                        break
                    orig_idx = current_front[idx]
                    new_pop.append(combined_pop[orig_idx])
                    new_fit.append(combined_fit[orig_idx])

                front_idx += 1

            population = np.array(new_pop, dtype=np.float64)
            fitness = np.array(new_fit, dtype=np.float64)

            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"第 {gen + 1:3d} 代: 外部归档集大小 = {len(self.archive)}")

        # 最终返回浮点格式的帕累托解
        return np.array(self.archive, dtype=np.float64), np.array(self.archive_fitness)

# ================= 主程序 =================
if __name__ == "__main__":
    # 设置随机种子但注释掉，保证每次运行结果不同
    # np.random.seed(42)
    np.set_printoptions(precision=3)

    # 创建一些初始基本解（示例, 包含8 个变量）
    def create_initial_solutions():
        """创建一些初始解作为示例"""
        # [Z1, Z2, Z3, Z5, Z6, phi_d1, phi_d2,m]
        initial_solutions = [
            [77.0, 127.0, 47.0, 25.0, 40.0, 0.2, 0.2, 2],
            [50.0, 92.0, 26.0, 21.0, 33.0, 0.2, 0.2,2],
            [49.0, 91.0, 25.0, 21.0, 33.0, 0.2, 0.2,2],
            [48.0, 90.0, 24.0, 21.0, 33.0, 0.2, 0.2,2],
            [47.0, 89.0, 23.0, 21.0, 33.0, 0.2, 0.2,2],
            [51.0, 87.0, 35.0, 18.0, 26.0, 0.2, 0.2,2],
            [50.0, 86.0, 34.0, 18.0, 26.0, 0.2, 0.2, 2],
            [45.0, 85.0, 23.0, 20.0, 31.0, 0.2, 0.2, 2],
            [40.0, 82.0, 18.0, 21.0, 32.0, 0.2, 0.2, 2],
            [42.0, 80.0, 22.0, 19.0, 29.0, 0.2, 0.2, 2],
            [34.0, 70.0, 18.0, 18.0, 26.0, 0.2, 0.2, 2],
            [33.0, 69.0, 19.0, 18.0, 25.0, 0.2, 0.2, 2],
            [30.0, 66.0, 18.0, 18.0, 24.0, 0.2, 0.2, 2],
            [28.0, 64.0, 18.0, 18.0, 23.0, 0.2, 0.2, 2],
            [26.0, 62.0, 18.0, 18.0, 22.0, 0.2, 0.2, 2],
            [24.0, 60.0, 18.0, 18.0, 21.0, 0.2, 0.2, 2],
            [22.0, 58.0, 18.0, 18.0, 20.0, 0.2, 0.2, 2],
            [21.0, 57.0, 19.0, 18.0, 19.0, 0.2, 0.2, 2]
        ]
        return initial_solutions

    # 获取初始解
    initial_solutions = create_initial_solutions()

    # 创建优化器（显式设置 n_var=8）
    nsga2 = NSGA2(
        fitness_func=fitness0,
        n_var=8,  # 变量总数调整为 8
        pop_size=30,
        n_gen=100,
        archive_size=50,
        crossover_prob=0.8,
        mutation_prob=0.2,
        initial_archive=initial_solutions  # 传入初始归档集
    )

    print(f"\n开始优化 (变量总数: {nsga2.n_var})")

    # 运行优化
    pareto_solutions, pareto_fitness = nsga2.evolve()

    # 输出结果
    if len(pareto_solutions) > 0:
        sorted_idx = np.argsort(pareto_fitness[:, 0])
        pareto_solutions = pareto_solutions[sorted_idx]
        pareto_fitness = pareto_fitness[sorted_idx]

        print(f"\n最终外部归档集中的帕累托最优解数量: {len(pareto_solutions)}")
        print("-" * 100)

        # 输出所有解（整数格式）并验证约束
        for idx in range(len(pareto_solutions)):
            ind = pareto_solutions[idx]
            f1, f2 = pareto_fitness[idx]
            efficiency = (1 - f1) * 100

            # 提取齿数参数并验证约束
            Z1, Z2, Z3, Z5, Z6 = ind[:5]
            constraints_ok = nsga2.check_gear_constraints(Z1, Z2, Z3, Z5, Z6)

            print(f"第{idx + 1}个前沿解：", ind, "目标值: ", [f1, f2])

        # 绘制帕累托前沿
        plt.figure(figsize=(10, 6))

        # 黑色连线
        plt.plot(pareto_fitness[:, 0], pareto_fitness[:, 1], 'k-', linewidth=2, alpha=0.8)
        # 红色圆圈标注
        plt.scatter(pareto_fitness[:, 0], pareto_fitness[:, 1],
                    s=80, c='red', edgecolors='black', linewidth=1.5,
                    marker='o', zorder=5)

        plt.xlabel("f1:1-效率", fontsize=12)
        plt.ylabel("f2:重量[N]", fontsize=12)
        plt.title("行星齿轮系统帕累托前沿", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

        # # 将输出结果保存为csv格式
        # if len(pareto_solutions) > 0:
        #     # 1. 获取当前运行脚本的绝对目录
        #     current_dir = os.path.dirname(os.path.abspath(__file__))
        #     # 2. 拼接文件路径
        #     filename = os.path.join(current_dir, "pareto_results.csv")
        #
        #     # 定义表头
        #     header = ['Index', 'Z1', 'Z2', 'Z3', 'Z5', 'Z6', 'phi_d1', 'phi_d2', 'f1_Loss', 'f2_Weight_N']
        #
        #     try:
        #         with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
        #             writer = csv.writer(f)
        #             writer.writerow(header)
        #
        #             for i in range(len(pareto_solutions)):
        #                 sol = pareto_solutions[i]
        #                 fit = pareto_fitness[i]
        #                 row = [i + 1] + list(sol) + list(fit)
        #                 writer.writerow(row)
        #
        #         print(f"\n[写入成功]")
        #         print(f"文件已保存至代码同级目录: {filename}")
        #
        #     except Exception as e:
        #         print(f"\n[写入失败] 错误原因: {e}")


        # 将输出结果保存为 Excel 格式 (.xlsx)
        if len(pareto_solutions) > 0:
            import pandas as pd

            # 1. 获取当前运行脚本的绝对目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filename_xlsx = os.path.join(current_dir, "pareto_results.xlsx")

            # 2. 准备数据
            header = ['Z1', 'Z2', 'Z3', 'Z5', 'Z6', 'phi_d1', 'phi_d2', 'm_index']
            df = pd.DataFrame(pareto_solutions, columns=header)

            # 添加目标函数值
            df['f1_Loss'] = pareto_fitness[:, 0]
            df['f2_Weight_N'] = pareto_fitness[:, 1]

            # 添加效率列 (可选，方便查看)
            df['Efficiency_%'] = (1 - df['f1_Loss']) * 100

            # 3. 写入 Excel
            try:
                # 使用 index=True 保留解的序号，或者 index_label='Index'
                df.to_excel(filename_xlsx, index=True, index_label='Rank', engine='openpyxl')
                print(f"\n[Excel 写入成功]")
                print(f"文件已保存至: {filename_xlsx}")
            except Exception as e:
                print(f"\n[Excel 写入失败] 错误原因: {e}")

        # 输出统计信息
        print(f"\n结果统计:")
        print(f"f1范围: [{pareto_fitness[:, 0].min():.6f}, {pareto_fitness[:, 0].max():.6f}]")
        print(f"f2范围: [{pareto_fitness[:, 1].min():.6f}, {pareto_fitness[:, 1].max():.6f}]")
        print(f"帕累托解数量: {len(pareto_solutions)},分别是：")
        # 输出符合格式的前沿，便于复制
        pareto_solutions_list = pareto_solutions.tolist()
        for i, sol in enumerate(pareto_solutions_list):
            indent = "    " if i < len(pareto_solutions_list) - 1 else "    "
            end_char = "," if i < len(pareto_solutions_list) - 1 else ""
            print(f"{indent}{sol}{end_char}")

    else:
        print("\n未找到帕累托最优解！")
