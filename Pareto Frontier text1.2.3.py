import numpy as np
import math
import csv  # 修改: 使用标准库 csv，不需要安装 pandas
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize


class PlanetaryGearOptimization(Problem):
    def __init__(self, m=2, rho=7800, g=9.81, fm=0.06):
        """
        行星齿轮优化问题
        设计变量 (9个): [Z1, Z2, Z3, Z5, Z6, b2, b5, b6, h]
        约束: b1=b5, b3=b6, b2 > b5+h+b6
        """
        self.m = m
        self.rho = rho
        self.g = g
        self.fm = fm

        # 变量数量: 9个 (b1和b3由其他变量决定，不参与独立优化)
        n_var = 9

        # 下界: [Z1, Z2, Z3, Z5, Z6, b2, b5, b6, h]
        xl = np.array([18, 18, 18, 18, 18, 10, 10, 10, 10])

        # 上界: b2上限设大一点，以容纳 b5+h+b6
        xu = np.array([100, 150, 100, 50, 40, 150, 40, 40, 40])

        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=10, xl=xl, xu=xu)

    # ================= 基础计算公式 =================
    def calculate_alpha_6(self, z_6):
        return math.acos((z_6 / (z_6 + 2)) * math.cos(np.radians(20)))

    def calculate_alpha_5(self, z_5):
        return math.acos(-z_5 / (z_5 + 2) * math.cos(np.radians(20)))

    def calculate_alpha_1(self, z_1):
        return math.acos(z_1 / (z_1 + 2) * math.cos(np.radians(20)))

    def calculate_alpha_2(self, z_2):
        return math.acos(-z_2 / (z_2 + 2) * math.cos(np.radians(20)))

    def calculate_alpha_3(self, z_3):
        return math.acos(z_3 / (z_3 + 2) * math.cos(np.radians(20)))

    # 重合度计算
    def calculate_epsilon_62(self, z_6, z_2, alpha_6, alpha, alpha_2):
        return (z_6 * (math.tan(alpha_6) - math.tan(alpha)) - z_2 * (math.tan(alpha) - math.tan(alpha_2))) / (
                    2 * math.pi)

    def calculate_epsilon_51(self, z_5, z_1, alpha_5, alpha, alpha_1):
        return (z_5 * (math.tan(alpha_5) - math.tan(alpha)) + z_1 * (math.tan(alpha_1) - math.tan(alpha))) / (
                    2 * math.pi)

    def calculate_epsilon_63(self, z_6, z_3, alpha_6, alpha, alpha_3):
        return (z_6 * (math.tan(alpha_6) - math.tan(alpha)) + z_3 * (math.tan(alpha_3) - math.tan(alpha))) / (
                    2 * math.pi)

    def calculate_epsilon_52(self, z_5, z_2, alpha_5, alpha, alpha_2):
        return (z_5 * (math.tan(alpha_5) - math.tan(alpha)) - z_2 * (math.tan(alpha_2) - math.tan(alpha))) / (
                    2 * math.pi)

    # Lambda计算
    def calculate_lambda1(self, epsilon_62, f_m, z_6, z_2):
        return (math.pi / 2) * epsilon_62 * f_m * (1 / z_6 - 1 / z_2)

    def calculate_lambda2(self, epsilon_52, f_m, z_5, z_2):
        return (math.pi / 2) * epsilon_52 * f_m * (1 / z_5 - 1 / z_2)

    def calculate_lambda3(self, epsilon_63, f_m, z_6, z_3):
        return (math.pi / 2) * epsilon_63 * f_m * (1 / z_6 + 1 / z_3)

    def calculate_lambda4(self, epsilon_51, f_m, z_5, z_1):
        return (math.pi / 2) * epsilon_51 * f_m * (1 / z_5 + 1 / z_1)

    def calculate_efficiency(self, Z1, Z2, Z3, Z5, Z6):
        try:
            alpha = np.radians(20)
            a1, a2, a3, a5, a6 = self.calculate_alpha_1(Z1), self.calculate_alpha_2(Z2), self.calculate_alpha_3(
                Z3), self.calculate_alpha_5(Z5), self.calculate_alpha_6(Z6)

            e62 = self.calculate_epsilon_62(Z6, Z2, a6, alpha, a2)
            e51 = self.calculate_epsilon_51(Z5, Z1, a5, alpha, a1)
            e63 = self.calculate_epsilon_63(Z6, Z3, a6, alpha, a3)
            e52 = self.calculate_epsilon_52(Z5, Z2, a5, alpha, a2)

            L1 = self.calculate_lambda1(e62, self.fm, Z6, Z2)
            L2 = self.calculate_lambda2(e52, self.fm, Z5, Z2)
            L3 = self.calculate_lambda3(e63, self.fm, Z6, Z3)
            L4 = self.calculate_lambda4(e51, self.fm, Z5, Z1)

            k = Z1 * (Z2 + Z3) / (Z2 * (Z1 - Z3))
            j = Z1 / (Z1 - Z3)

            num = (((1 - L1) * L2 + L1) * (1 - L4) + L3 * (1 - L1) * (1 - L2 * (1 - L4) - L4) + L4) * (
                        1 - (L3 * (1 - L1) + L1) * (1 - j / k))
            den = (1 / (j - 1) + L4 + L2 * (1 - L4) + (L3 * (1 - L1) + L1) * (1 - L2 * (1 - L4) - L4))

            efficiency = 1 - abs((num / den) + (L1 + L3 * (1 - L1)) * (1 - j / k))
            return efficiency
        except:
            return 0.9

    def calculate_volume(self, Z1, Z2, Z3, Z5, Z6, b1, b2, b3, b5, b6, h):
        try:
            d1, d3, d5, d6 = self.m * Z1, self.m * Z3, self.m * Z5, self.m * Z6
            d2a = self.m * (Z2 + 2.0)
            d2f = self.m * (Z2 - 2.5)

            V_sun = (np.pi / 4) * (b1 * (d1 ** 2) + b3 * (d3 ** 2))
            V_ring = (np.pi / 4) * b2 * (d2a ** 2 - d2f ** 2)
            V_planet = (np.pi / 4) * 3 * (b5 * (d5 ** 2) + b6 * (d6 ** 2))
            V_carrier = np.pi * (d6 / 2) ** 2 * h

            return (V_sun + V_ring + V_planet + V_carrier) * 1e-9
        except:
            return 1e-3

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        f1, f2 = np.zeros(n_pop), np.zeros(n_pop)
        g = np.zeros((n_pop, 10))

        for i in range(n_pop):
            Z1, Z2, Z3, Z5, Z6 = [int(round(x)) for x in X[i, :5]]
            b2, b5, b6, h = X[i, 5:]
            b1, b3 = b5, b6

            # 目标函数
            f1[i] = 1 - self.calculate_efficiency(Z1, Z2, Z3, Z5, Z6)
            f2[i] = self.rho * self.calculate_volume(Z1, Z2, Z3, Z5, Z6, b1, b2, b3, b5, b6, h) * self.g

            # 约束条件
            g[i, 0] = (Z3 + Z6) - (Z1 + Z5) + 0.1
            g[i, 1] = Z3 - Z1 + 0.1
            g[i, 2] = Z5 - Z6 + 0.1
            g[i, 3] = max(abs((Z1 + 2 * Z5) - Z2), abs((Z3 + 2 * Z6) - Z2)) - 1

            d1, d3, d5, d6 = self.m * Z1, self.m * Z3, self.m * Z5, self.m * Z6
            psis = [b1 / d1, b3 / d3, b5 / d5, b6 / d6]
            g[i, 4] = 0.3 - min(psis)
            g[i, 5] = max(psis) - 0.8

            # **核心约束**: b2 > b5 + h + b6
            g[i, 6] = (b5 + h + b6) - b2 + 0.5
            g[i, 7] = max(b5, b6) - h + 0.5
            g[i, 8], g[i, 9] = 0, 0

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g


# ================= 优化执行 =================
try:
    problem = PlanetaryGearOptimization(m=2, rho=7800, g=9.81, fm=0.06)

    algorithm = NSGA2(
        pop_size=1000,  # 种群 500
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    print("开始优化...")
    print("配置: 种群规模=500, 迭代次数=500")
    print("约束: b1=b5, b3=b6, b2 > b5+h+b6")

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 1000),  # 迭代 500
                   verbose=True)

    if res.X is not None and len(res.X) > 0:
        print("\n优化完成! 正在处理数据...")

        # 4. 数据处理与导出 CSV (使用内置 csv 模块，替代 pandas)
        results = []
        for i in range(len(res.X)):
            Z1, Z2, Z3, Z5, Z6 = [int(round(x)) for x in res.X[i, :5]]
            b2, b5, b6, h = res.X[i, 5:]
            b1, b3 = b5, b6

            loss = res.F[i, 0]
            weight = res.F[i, 1]
            efficiency = 1 - loss
            volume = problem.calculate_volume(Z1, Z2, Z3, Z5, Z6, b1, b2, b3, b5, b6, h) * 1e9
            width_check = "Pass" if b2 > (b5 + h + b6) else "Fail"

            # 字典用于后续写入
            row = {
                "ID": i + 1,
                "Z1": Z1, "Z2": Z2, "Z3": Z3, "Z5": Z5, "Z6": Z6,
                "b1 (mm)": round(b1, 2),
                "b2 (mm)": round(b2, 2),
                "b3 (mm)": round(b3, 2),
                "b5 (mm)": round(b5, 2),
                "b6 (mm)": round(b6, 2),
                "h (mm)": round(h, 2),
                "Efficiency": round(efficiency, 5),
                "Efficiency_Loss": round(loss, 5),
                "Weight (N)": round(weight, 2),
                "Volume (mm^3)": round(volume, 0),
                "Constraint_Check": width_check
            }
            results.append(row)

        # 导出 CSV
        csv_filename = "pareto_optimization_results.csv"
        headers = ["ID", "Z1", "Z2", "Z3", "Z5", "Z6",
                   "b1 (mm)", "b2 (mm)", "b3 (mm)", "b5 (mm)", "b6 (mm)", "h (mm)",
                   "Efficiency", "Efficiency_Loss", "Weight (N)", "Volume (mm^3)", "Constraint_Check"]

        with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n成功! 结果已保存至: {csv_filename}")

    else:
        print("未找到可行解。")

except Exception as e:
    print(f"运行出错: {e}")
    import traceback

    traceback.print_exc()