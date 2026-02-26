import itertools
import csv


class PGTNamingChainGenerator:
    def __init__(self):
        self.nc = 1
        # 合法极性对 (Sm, Pm): W-W, W-N, N-W
        self.legal_mesh_pairs = [('W', 'W'), ('W', 'N'), ('N', 'W')]

    def validate_constraints(self, n, ns, np):
        """执行 text8.py 拓扑约束"""
        if ns + self.nc < 3: return False
        if 2 * np > n - 2: return False
        if 2 * np < n - 1 - 2 * self.nc: return False
        if self.nc > np: return False
        return True

    def generate(self, n_range=[4, 5, 6]):
        final_data = []
        for n in n_range:
            for ns in range(1, n - 1):
                np_val = n - ns - self.nc
                if not self.validate_constraints(n, ns, np_val):
                    continue

                # 设定总啮合节数 K，通常 4-6 构件下 K = n-2 或 n-3
                # 为了覆盖 Sg, Pg 逻辑，我们遍历可能的总节数 K
                for K in range(max(ns, np_val), n - 1):
                    # 分配 Sg: 将 K 分解为 ns 个正整数的和
                    s_sg_combos = self._get_sum_partitions(K, ns)
                    # 分配 Pg: 将 K 分解为 np 个正整数的和
                    p_pg_combos = self._get_sum_partitions(K, np_val)

                    for s_sgs in s_sg_combos:
                        for p_pgs in p_pg_combos:
                            # 构造基础构件序列
                            sun_seq = []
                            for i, count in enumerate(s_sgs):
                                sun_seq.extend([f"S{i + 1}"] * count)

                            planet_seq = []
                            for j, count in enumerate(p_pgs):
                                planet_seq.extend([f"P{j + 1}"] * count)

                            # 枚举极性组合
                            for polarities in itertools.product(self.legal_mesh_pairs, repeat=K):
                                chain_parts = []
                                for idx in range(K):
                                    sm, pm = polarities[idx]
                                    chain_parts.append(f"{sun_seq[idx]}{sm}-{planet_seq[idx]}{pm}")

                                naming_chain = "-".join(chain_parts) + "-C1"

                                # 填充行数据
                                row = {
                                    "n": n, "ns": ns, "np": np_val,
                                    "Naming_Chain": naming_chain
                                }
                                # 动态添加 si_sg 与 pj_pg 列
                                for i in range(ns): row[f"s{i + 1}_sg"] = s_sgs[i]
                                for j in range(np_val): row[f"p{j + 1}_pg"] = p_pgs[j]

                                final_data.append(row)
        return final_data

    def _get_sum_partitions(self, total, count):
        """将 total 分解为 count 个正整数的和"""
        res = []
        for p in itertools.product(range(1, total + 1), repeat=count):
            if sum(p) == total:
                res.append(list(p))
        return res

    def save_csv(self, data, filename="PGT_Naming_Detailed_456.csv"):
        if not data: return
        # 获取所有可能的列名（考虑不同 ns, np 产生的列）
        fieldnames = ["n", "ns", "np"]
        max_ns = max(d['ns'] for d in data)
        max_np = max(d['np'] for d in data)
        for i in range(1, max_ns + 1): fieldnames.append(f"s{i}_sg")
        for j in range(1, max_np + 1): fieldnames.append(f"p{j}_pg")
        fieldnames.append("Naming_Chain")

        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        print(f"表格生成完成，共 {len(data)} 行数据。")


# 执行
engine = PGTNamingChainGenerator()
data = engine.generate([4, 5, 6])
engine.save_csv(data)