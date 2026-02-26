import numpy as np
import math
import matplotlib.pyplot as plt
import random

# ========== 修复中文显示乱码问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 优先使用无衬线字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def fitness0(individual):
    """
    计算行星齿轮系统的两个目标函数值
    输入本身就是整数，无需强制转换
    """
    # ================= 提取设计变量（输入保证为整数） =================
    m = 2  # 模数（整数）
    rho = 7800  # 材料密度 (kg/m³)
    g = 9.81  # 重力加速度 (m/s²)
    fm = 0.06  # 摩擦系数

    # 提取整数变量（输入已保证为整数）
    Z1, Z2, Z3, Z5, Z6 = individual[:5]
    b2, b5, b6, h = individual[5:]

    # 约束：b1=b5, b3=b6（整数）
    b1 = b5
    b3 = b6

    # 关键约束检查（整数运算）
    # 齿数约束在个体生成阶段已保证，此处仅做最终验证
    if Z1 == Z3:
        Z1 = Z1 + 1 if Z1 < 100 else Z1 - 1  # 整数调整
    if b2 <= b5 + h + b6:
        b2 = (b5 + h + b6) + 1  # 整数调整满足约束

    # ================= 内部辅助函数 =================
    def calculate_alpha_6(z_6):
        cos_val = (z_6 / (z_6 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_5(z_5):
        cos_val = (-z_5 / (z_5 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_1(z_1):
        cos_val = (z_1 / (z_1 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_2(z_2):
        cos_val = (-z_2 / (z_2 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    def calculate_alpha_3(z_3):
        cos_val = (z_3 / (z_3 + 2)) * math.cos(math.radians(20))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.acos(cos_val)

    # ================= 效率计算 (f1 = 1 - 效率) =================
    alpha = math.radians(20)
    a1 = calculate_alpha_1(Z1)
    a2 = calculate_alpha_2(Z2)
    a3 = calculate_alpha_3(Z3)
    a5 = calculate_alpha_5(Z5)
    a6 = calculate_alpha_6(Z6)

    # 计算epsilon值（核心效率参数）
    e62 = (Z6 * (math.tan(a6) - math.tan(alpha)) - Z2 * (math.tan(alpha) - math.tan(a2))) / (2 * math.pi)
    e51 = (Z5 * (math.tan(a5) - math.tan(alpha)) + Z1 * (math.tan(a1) - math.tan(alpha))) / (2 * math.pi)
    e63 = (Z6 * (math.tan(a6) - math.tan(alpha)) + Z3 * (math.tan(a3) - math.tan(alpha))) / (2 * math.pi)
    e52 = (Z5 * (math.tan(a5) - math.tan(alpha)) - Z2 * (math.tan(a2) - math.tan(alpha))) / (2 * math.pi)

    # 计算lambda值
    lambda1 = (math.pi / 2) * e62 * fm * (1 / Z6 - 1 / Z2) if Z6 != Z2 else 0.0
    lambda2 = (math.pi / 2) * e52 * fm * (1 / Z5 - 1 / Z2) if Z5 != Z2 else 0.0
    lambda3 = (math.pi / 2) * e63 * fm * (1 / Z6 + 1 / Z3)
    lambda4 = (math.pi / 2) * e51 * fm * (1 / Z5 + 1 / Z1)

    # 计算k和j
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
        efficiency = 0.7  # 默认值

    # 限制效率范围
    efficiency = np.clip(efficiency, 0.50001, 0.98999)
    f1 = 1.0 - efficiency  # 目标1：1-效率（越小越好）

    # ================= 重量计算 (f2) =================
    # 计算各部分直径（整数运算）
    d1 = m * Z1
    d3 = m * Z3
    d5 = m * Z5
    d6 = m * Z6
    d2a = m * (Z2 + 2)
    d2f = max(1, m * (Z2 - 2))  # 确保正数

    # 计算各部分体积（整数运算）
    V_sun = (math.pi / 4) * (b1 * (d1 ** 2) + b3 * (d3 ** 2))
    V_ring = (math.pi / 4) * b2 * (d2a ** 2 - d2f ** 2)
    V_planet = (math.pi / 4) * 3 * (b5 * (d5 ** 2) + b6 * (d6 ** 2))
    V_carrier = math.pi * (d6 // 2) ** 2 * h

    # 总体积（m³）
    total_volume = (V_sun + V_ring + V_planet + V_carrier) * 1e-9

    # 重量计算（kg → N）
    mass = total_volume * rho
    f2 = mass * g  # 目标2：重量（越小越好）

    return f1, f2


class NSGA2:
    def __init__(self,
                 fitness_func,
                 n_var=9,
                 n_obj=2,
                 pop_size=100,
                 n_gen=200,
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

        # 变量上下界（强制整数）
        if lower_bounds is None:
            self.lower_bounds = np.array([18, 18, 18, 18, 18, 10, 10, 10, 10], dtype=np.int64)
        else:
            self.lower_bounds = np.array(lower_bounds, dtype=np.int64)

        if upper_bounds is None:
            self.upper_bounds = np.array([100, 150, 100, 50, 40, 150, 40, 40, 40], dtype=np.int64)
        else:
            self.upper_bounds = np.array(upper_bounds, dtype=np.int64)

        # 外部归档集 - 只保存非支配解（整数）
        self.archive = []
        self.archive_fitness = []

        # 初始化外部归档集（如果提供了初始解）
        if initial_archive is not None:
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
            initial_archive: 初始解列表或数组，每个解是一个9维整数向量
        """
        if len(initial_archive) == 0:
            return

        # 确保初始归档集是numpy数组格式
        initial_archive = np.array(initial_archive, dtype=np.int64)

        # 计算所有初始解的适应度
        initial_fitness = []
        for ind in initial_archive:
            try:
                f1, f2 = self.fitness_func(ind)
                initial_fitness.append([f1, f2])
            except Exception as e:
                print(f"警告：计算初始解适应度时出错: {e}")
                # 如果计算失败，跳过该解
                continue

        if len(initial_fitness) == 0:
            print("警告：所有初始解都计算适应度失败，归档集为空")
            return

        initial_fitness = np.array(initial_fitness, dtype=np.float64)

        # 1. 非支配排序，找出所有非支配解（第一前沿）
        fronts, _ = self.fast_non_dominated_sort(initial_fitness)

        if len(fronts) == 0:
            print("警告：初始归档集非支配排序失败")
            return

        first_front = fronts[0]  # 第一前沿包含所有非支配解

        # 2. 将所有第一前沿解加入归档集
        for idx in first_front:
            # 检查是否为有效解
            if np.all(np.isfinite(initial_fitness[idx])):
                # 检查是否被已有归档集解支配
                is_dominated = False
                for arch_fit in self.archive_fitness:
                    if self.dominates(arch_fit, initial_fitness[idx]):
                        is_dominated = True
                        break

                if not is_dominated:
                    # 移除被当前解支配的归档集解
                    to_remove = []
                    for i, arch_fit in enumerate(self.archive_fitness):
                        if self.dominates(initial_fitness[idx], arch_fit):
                            to_remove.append(i)

                    for i in sorted(to_remove, reverse=True):
                        del self.archive[i]
                        del self.archive_fitness[i]

                    # 检查是否重复
                    is_duplicate = False
                    for arch_ind in self.archive:
                        if np.array_equal(arch_ind, initial_archive[idx]):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        self.archive.append(initial_archive[idx])
                        self.archive_fitness.append(initial_fitness[idx])

        # 3. 如果归档集超过大小限制，按拥挤距离裁剪
        if len(self.archive) > self.archive_size:
            self.trim_archive()

        # print(f"初始化外部归档集完成：原始解数量={len(initial_archive)}，非支配解数量={len(self.archive)}")

    def add_to_archive(self, individual, fitness):
        """单个个体加入归档集（确保个体为整数）"""
        # 检查是否被归档集支配
        is_dominated = False
        for arch_fit in self.archive_fitness:
            if self.dominates(arch_fit, fitness):
                is_dominated = True
                break

        if is_dominated:
            return False

        # 找出被当前个体支配的归档集元素
        to_remove = []
        for idx, arch_fit in enumerate(self.archive_fitness):
            if self.dominates(fitness, arch_fit):
                to_remove.append(idx)

        # 删除被支配的元素
        for idx in sorted(to_remove, reverse=True):
            del self.archive[idx]
            del self.archive_fitness[idx]

        # 检查是否重复（整数比较）
        is_duplicate = False
        for arch_ind in self.archive:
            if np.array_equal(arch_ind, individual):
                is_duplicate = True
                break

        if not is_duplicate:
            self.archive.append(individual)
            self.archive_fitness.append(fitness)

        # 裁剪归档集
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
        """移除种群中的重复个体，保持种群多样性"""
        # 转换为元组以便去重
        unique_tuples = []
        unique_indices = []

        for idx, ind in enumerate(population):
            ind_tuple = tuple(ind)
            if ind_tuple not in unique_tuples:
                unique_tuples.append(ind_tuple)
                unique_indices.append(idx)

        # 如果去重后数量不足，补充新的随机个体
        unique_pop = population[unique_indices]
        if len(unique_pop) < len(population):
            need_ = len(population) - len(unique_pop)
            new_individuals = []
            for _ in range(need_):
                new_individuals.append(self.create_valid_individual())
            new_individuals = np.array(new_individuals, dtype=np.int64)
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

        # 边界约束
        constraint6 = (self.lower_bounds[0] <= Z1 <= self.upper_bounds[0])
        constraint7 = (self.lower_bounds[1] <= Z2 <= self.upper_bounds[1])
        constraint8 = (self.lower_bounds[2] <= Z3 <= self.upper_bounds[2])
        constraint9 = (self.lower_bounds[3] <= Z5 <= self.upper_bounds[3])
        constraint10 = (self.lower_bounds[4] <= Z6 <= self.upper_bounds[4])

        return (constraint1 and constraint2 and constraint3 and
                constraint4 and constraint5 and constraint6 and
                constraint7 and constraint8 and constraint9 and constraint10)

    def generate_valid_gear_teeth(self):
        """生成满足所有齿数约束的齿轮参数（修复边界条件）"""
        max_attempts = 1000
        for _ in range(max_attempts):
            # 1. 先确定Z5和Z6，确保Z5 < Z6且在边界内
            # Z5的上限为Z6上限-1，避免Z5+1超过Z6上限
            z5_max = min(self.upper_bounds[3], self.upper_bounds[4] - 1)
            if z5_max <= self.lower_bounds[3]:
                z5_max = self.lower_bounds[3]  # 保底

            Z5 = np.random.randint(self.lower_bounds[3], z5_max + 1)
            # Z6必须大于Z5且不超过上限
            Z6 = np.random.randint(Z5 + 1, self.upper_bounds[4] + 1)

            # 2. 确定Z3和Z1，确保Z1 > Z3且在边界内
            # Z3的上限为Z1上限-1
            z3_max = min(self.upper_bounds[2], self.upper_bounds[0] - 1)
            if z3_max <= self.lower_bounds[2]:
                z3_max = self.lower_bounds[2]  # 保底

            Z3 = np.random.randint(self.lower_bounds[2], z3_max + 1)
            Z1 = np.random.randint(Z3 + 1, self.upper_bounds[0] + 1)

            # 3. 根据约束计算Z2，并检查Z2边界
            Z2_from_Z1 = Z1 + 2 * Z5
            Z2_from_Z3 = Z3 + 2 * Z6

            # 确保两个方式计算的Z2一致且在边界内
            if Z2_from_Z1 == Z2_from_Z3:
                Z2 = Z2_from_Z1
                # 检查Z2边界和Z1+Z5 > Z3+Z6约束
                if (self.lower_bounds[1] <= Z2 <= self.upper_bounds[1] and
                        (Z1 + Z5) > (Z3 + Z6)):
                    # 验证所有约束
                    if self.check_gear_constraints(Z1, Z2, Z3, Z5, Z6):
                        return Z1, Z2, Z3, Z5, Z6

        # 如果多次尝试失败，返回安全的默认值（确保满足所有约束）
        # 默认值：Z1=20, Z2=60, Z3=18, Z5=20, Z6=21
        return 20, 60, 18, 20, 21

    def create_valid_individual(self):
        """创建有效整数个体（满足所有齿轮约束）"""
        max_attempts = 100
        for _ in range(max_attempts):
            # 生成满足齿数约束的齿轮参数
            Z1, Z2, Z3, Z5, Z6 = self.generate_valid_gear_teeth()

            # 生成其他参数（b2, b5, b6, h）
            b5 = np.random.randint(self.lower_bounds[6], self.upper_bounds[6] + 1)  # 修正索引
            b6 = np.random.randint(self.lower_bounds[7], self.upper_bounds[7] + 1)  # 修正索引
            h = np.random.randint(self.lower_bounds[8], self.upper_bounds[8] + 1)  # 修正索引
            b2 = (b5 + h + b6) + 1  # 确保b2 > b5 + h + b6

            # 确保b2在边界内
            b2 = np.clip(b2, self.lower_bounds[5], self.upper_bounds[5])  # 修正索引

            # 构建完整个体
            individual = np.array([Z1, Z2, Z3, Z5, Z6, b2, b5, b6, h], dtype=np.int64)

            # 验证个体有效性
            try:
                f1, f2 = self.fitness_func(individual)
                if np.isfinite(f1) and np.isfinite(f2):
                    return individual
            except Exception as e:
                print(f"创建个体时出错: {e}")
                continue

        # 默认整数个体（满足所有约束）
        return np.array([20, 60, 18, 20, 21, 37, 15, 10, 10], dtype=np.int64)

    def initialize_population(self):
        """初始化整数种群（确保无重复）"""
        population = []
        for i in range(self.pop_size):
            population.append(self.create_valid_individual())
        population = np.array(population, dtype=np.int64)

        # 初始化时就移除重复个体
        population = self.remove_duplicates(population)

        # 评估初始种群
        fitness = self.evaluate_population(population)

        # 非支配排序，只取第一前沿
        fronts, _ = self.fast_non_dominated_sort(fitness)
        first_front = fronts[0] if fronts else []

        # 只将第一前沿解加入归档集
        for idx in first_front:
            self.add_to_archive(population[idx], fitness[idx])

        # print(f"初始化完成：种群大小={len(population)}, 归档集大小={len(self.archive)}")
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
        """整数交叉操作（保证交叉后仍满足齿轮约束）"""
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        # 先进行基础交叉
        child1 = parent1.copy()
        child2 = parent2.copy()

        # 只交叉非齿数参数或特殊处理齿数参数
        # 齿数参数(Z1,Z2,Z3,Z5,Z6)需要保持约束，单独处理
        # 仅交叉结构参数(b2,b5,b6,h)
        cross_points = np.random.choice([5, 6, 7, 8], size=2, replace=False)
        for point in cross_points:
            child1[point], child2[point] = child2[point], child1[point]

        # 修复结构参数约束
        child1 = self.repair_individual(child1)
        child2 = self.repair_individual(child2)

        # 验证并修复齿数约束（如果被破坏）
        for child in [child1, child2]:
            Z1, Z2, Z3, Z5, Z6 = child[:5]
            if not self.check_gear_constraints(Z1, Z2, Z3, Z5, Z6):
                # 重新生成满足约束的齿数参数
                new_Z1, new_Z2, new_Z3, new_Z5, new_Z6 = self.generate_valid_gear_teeth()
                child[:5] = [new_Z1, new_Z2, new_Z3, new_Z5, new_Z6]

        return child1, child2

    def mutation(self, individual):
        """整数变异操作（保证变异后仍满足齿轮约束）"""
        mutated = individual.copy()

        # 分别处理齿数参数和结构参数
        for i in range(self.n_var):
            if np.random.random() < self.mutation_prob:
                if i < 5:  # 齿数参数，特殊处理
                    # 重新生成整个齿数组合以保持约束
                    Z1, Z2, Z3, Z5, Z6 = self.generate_valid_gear_teeth()
                    mutated[:5] = [Z1, Z2, Z3, Z5, Z6]
                    break  # 一次只变异一个齿数组合
                else:  # 结构参数，常规变异
                    mutation_step = np.random.choice([-3, -2, -1, 1, 2, 3])
                    mutated[i] += mutation_step
                    mutated[i] = np.clip(mutated[i], self.lower_bounds[i], self.upper_bounds[i])

        # 修复约束
        mutated = self.repair_individual(mutated)

        # 最终验证约束
        Z1, Z2, Z3, Z5, Z6 = mutated[:5]
        if not self.check_gear_constraints(Z1, Z2, Z3, Z5, Z6):
            new_Z1, new_Z2, new_Z3, new_Z5, new_Z6 = self.generate_valid_gear_teeth()
            mutated[:5] = [new_Z1, new_Z2, new_Z3, new_Z5, new_Z6]

        return mutated

    def repair_individual(self, individual):
        """修复整数个体约束（仅调整整数）"""
        # 修复结构参数约束
        b2, b5, b6, h = individual[5], individual[6], individual[7], individual[8]
        if b2 <= b5 + h + b6:
            individual[5] = (b5 + h + b6) + 1

        # 确保结构参数在边界内
        for i in range(5, 9):
            individual[i] = np.clip(individual[i], self.lower_bounds[i], self.upper_bounds[i])

        return individual

    def evolve(self):
        """主进化过程（增加多样性维护）"""
        # 初始化整数种群
        population = self.initialize_population()
        fitness = self.evaluate_population(population)

        for gen in range(self.n_gen):
            # 非支配排序
            fronts, _ = self.fast_non_dominated_sort(fitness)

            # 选择
            parents = self.selection(population, fitness, fronts)

            # 交叉变异生成子代（纯整数）
            offspring = []
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    p1 = parents[i]
                    p2 = parents[i + 1]
                    c1, c2 = self.crossover(p1, p2)
                    c1 = self.mutation(c1)
                    c2 = self.mutation(c2)
                    offspring.extend([c1, c2])

            # 确保子代数量
            while len(offspring) < self.pop_size:
                offspring.append(self.create_valid_individual())

            offspring = np.array(offspring[:self.pop_size], dtype=np.int64)

            # 移除子代中的重复个体（关键：防止种群同质化）
            offspring = self.remove_duplicates(offspring)

            offspring_fitness = self.evaluate_population(offspring)

            # 只将子代的第一前沿解加入归档集
            offspring_fronts, _ = self.fast_non_dominated_sort(offspring_fitness)
            offspring_first_front = offspring_fronts[0] if offspring_fronts else []
            for idx in offspring_first_front:
                self.add_to_archive(offspring[idx], offspring_fitness[idx])

            # 合并父代子代
            combined_pop = np.vstack([population, offspring])
            combined_fit = np.vstack([fitness, offspring_fitness])

            # 移除合并种群中的重复个体
            combined_pop = self.remove_duplicates(combined_pop)
            # 重新评估去重后的种群
            combined_fit = self.evaluate_population(combined_pop)

            # 非支配排序并选择新种群
            combined_fronts, _ = self.fast_non_dominated_sort(combined_fit)
            new_pop = []
            new_fit = []
            front_idx = 0

            while len(new_pop) < self.pop_size and front_idx < len(combined_fronts):
                current_front = combined_fronts[front_idx]
                if len(current_front) == 0:
                    front_idx += 1
                    continue

                # 计算当前前沿的拥挤距离
                front_fitness = combined_fit[current_front]
                distances = self.calculate_crowding_distance(front_fitness)

                # 按拥挤距离排序
                sorted_indices = np.argsort(-distances)

                # 选择个体
                for idx in sorted_indices:
                    if len(new_pop) >= self.pop_size:
                        break
                    orig_idx = current_front[idx]
                    new_pop.append(combined_pop[orig_idx])
                    new_fit.append(combined_fit[orig_idx])

                front_idx += 1

            # 更新种群（纯整数）
            population = np.array(new_pop, dtype=np.int64)
            fitness = np.array(new_fit, dtype=np.float64)

            # 每10代输出归档集大小
            if (gen + 1) % 10 == 0 or gen == 0:
                print(f"第 {gen + 1:3d} 代: 外部归档集大小 = {len(self.archive)}")

        # 最终返回整数格式的帕累托解
        return np.array(self.archive, dtype=np.int64), np.array(self.archive_fitness)


# ================= 主程序 =================
if __name__ == "__main__":
    # 设置随机种子但注释掉，保证每次运行结果不同
    # np.random.seed(42)
    np.set_printoptions(precision=3)


    # 创建一些初始基本解（示例）
    def create_initial_solutions():
        """创建一些初始解作为示例"""
        initial_solutions = [[28, 64, 18, 18, 23, 31, 10, 10, 10],
                             [26, 62, 18, 18, 22, 31, 10, 10, 10],
                             [24, 60, 18, 18, 21, 31, 10, 10, 10],
                             [22, 58, 18, 18, 20, 31, 10, 10, 10],
                             [20, 56, 18, 18, 19, 31, 10, 10, 10]]
        return initial_solutions


    # 获取初始解
    initial_solutions = create_initial_solutions()
    # print(f"创建了 {len(initial_solutions)} 个初始解用于初始化外部归档集")

    # 创建优化器（显式设置交叉率，并传入初始归档集）
    nsga2 = NSGA2(
        fitness_func=fitness0,
        pop_size=40,
        n_gen=500,
        archive_size=50,
        crossover_prob=0.8,
        mutation_prob=0.2,
        initial_archive=initial_solutions  # 传入初始归档集
    )

    print(f"\n开始优化")

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
