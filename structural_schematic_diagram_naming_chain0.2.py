# import itertools
# import csv
#
#
# class PGTNamingGenerator:
#     def __init__(self):
#         self.nc = 1  # 设定为单载体系统
#         self.mesh_types = ['W', 'N']
#
#     def check_topological_constraints(self, n, ns, np):
#         """实现 text8.py 中的 1-DOF PGT 拓扑约束"""
#         if ns + self.nc < 3: return False
#         if 2 * np > n - 2: return False
#         if 2 * np < n - 1 - 2 * self.nc: return False
#         if self.nc > np: return False
#         return True
#
#     def is_pseudo_name(self, chain):
#         """伪命名判别准则"""
#         forbidden = ["PN-SN", "SN-PN"]
#         return any(p in chain for p in forbidden)
#
#     def generate_all_chains(self, n_range=[4, 5, 6]):
#         all_results = []
#
#         for n in n_range:
#             # 1. 寻找满足约束的 ns, np 组合
#             for ns in range(1, n):
#                 np = n - ns - self.nc
#                 if self.check_topological_constraints(n, ns, np):
#                     # 2. 遍历齿轮段数 Sg <= ns, Pg <= np
#                     for sg in range(1, ns + 1):
#                         for pg in range(1, np + 1):
#                             # 生成啮合特征组合 (W/N)
#                             # 链条长度通常与段数相关
#                             combos = list(itertools.product(self.mesh_types, repeat=sg + pg))
#
#                             for combo in combos:
#                                 # 构造 S-P 节点序列
#                                 segments = []
#                                 for i in range(max(sg, pg)):
#                                     s_idx = (i % sg) + 1
#                                     p_idx = (i % pg) + 1
#                                     # 模拟 S-P 结构
#                                     s_part = f"S{s_idx}{combo[i % len(combo)]}"
#                                     p_part = f"P{p_idx}{combo[(i + 1) % len(combo)]}"
#                                     segments.append(f"{s_part}-{p_part}")
#
#                                 naming_chain = "-".join(segments) + "-C1"
#
#                                 # 3. 过滤伪命名
#                                 if not self.is_pseudo_name(naming_chain):
#                                     all_results.append({
#                                         "Total_Nodes_n": n,
#                                         "ns": ns,
#                                         "np": np,
#                                         "Sg_Segments": sg,
#                                         "Pg_Segments": pg,
#                                         "Naming_Chain": naming_chain
#                                     })
#         return all_results
#
#     def export_csv(self, data, filename="PGT_Naming_Chains_Final.csv"):
#         if not data: return
#         keys = data[0].keys()
#         with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
#             writer = csv.DictWriter(f, fieldnames=keys)
#             writer.writeheader()
#             writer.writerows(data)
#         print(f"数据处理完成。已生成 {len(data)} 条可行命名链，保存至: {filename}")
#
#
# # 执行生成任务
# generator = PGTNamingGenerator()
# chains_data = generator.generate_all_chains([4, 5, 6])
# generator.export_csv(chains_data)
import itertools
import csv


class PGTComponentIntegrityGenerator:
    """基于构件完整性约束的命名链生成引擎"""

    def __init__(self):
        self.nc = 1
        self.mesh_types = ['W', 'N']

    def validate_topology(self, n, ns, np):
        """执行 text8.py 中的 1-DOF PGT 拓扑约束"""
        if ns + self.nc < 3: return False
        if 2 * np > n - 2: return False
        if 2 * np < n - 1 - 2 * self.nc: return False
        if self.nc > np: return False
        return True

    def is_valid_sequence(self, chain):
        """伪命名判别"""
        return not any(p in chain for p in ["PN-SN", "SN-PN"])

    def generate_chains(self, n_total):
        results = []
        # 遍历所有可能的 ns, np 组合
        for ns in range(1, n_total):
            np_val = n_total - ns - self.nc
            if not self.validate_topology(n_total, ns, np_val):
                continue

            # 确定齿轮段数 (由于 Sg <= ns 且必须涵盖所有索引，此处设定段数范围)
            # 在 4-6 构件中，通常段数之和等于构件数
            sun_indices = [f"S{i + 1}" for i in range(ns)]
            planet_indices = [f"P{j + 1}" for j in range(np_val)]

            # 生成排列组合，确保所有构件索引至少出现一次
            # 简化逻辑：生成长度为 max(ns, np_val) * 2 的基本链
            length = max(ns, np_val)
            mesh_combos = list(itertools.product(self.mesh_types, repeat=length * 2))

            for combo in mesh_combos:
                chain_parts = []
                for i in range(length):
                    # 循环使用索引以确保覆盖，并添加啮合特征
                    s_id = sun_indices[i % ns]
                    p_id = planet_indices[i % np_val]

                    s_node = f"{s_id}{combo[2 * i]}"
                    p_node = f"{p_id}{combo[2 * i + 1]}"
                    chain_parts.append(f"{s_node}-{p_node}")

                naming_chain = "-".join(chain_parts) + "-C1"

                # 验证构件完整性：检查是否包含所有 S1..Sns 和 P1..Pnp
                all_s_present = all(s in naming_chain for s in sun_indices)
                all_p_present = all(p in naming_chain for p in planet_indices)

                if all_s_present and all_p_present and self.is_valid_sequence(naming_chain):
                    results.append({
                        "n": n_total,
                        "ns": ns,
                        "np": np_val,
                        "Naming_Chain": naming_chain
                    })
        return results

    def run_and_export(self, n_list=[4, 5, 6], filename="PGT_NamingChains_456.csv"):
        final_data = []
        for n in n_list:
            final_data.extend(self.generate_chains(n))

        if final_data:
            keys = final_data[0].keys()
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(final_data)
            print(f"成功导出 {len(final_data)} 条命名链至 {filename}")


# 执行生成
engine = PGTComponentIntegrityGenerator()
engine.run_and_export()