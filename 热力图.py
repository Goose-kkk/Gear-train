import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 原始代码
# # 读取原始数据
# data = pd.read_excel("pareto_results.xlsx", sheet_name="Sheet1")
# # 计算并绘制皮尔逊相关性矩阵（括号中不填计算模式，默认为皮尔逊相关性系数计算）
# # corr_matrix = data.corr()
# corr_matrix = data.T.corr()
# # corr_matrix = data.corr(method='spearman')
#
# # 生成热力图
# plt.rcParams["font.family"] = "SimHei"  # 中文黑体，使中文正确显示
# plt.rcParams['axes.unicode_minus'] = False  # 消除SimHei对符号的影响
#
# plt.figure(figsize=(16, 15))
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# # plt.title('相关系数热力图', fontsize=20)
#
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False, linewidths=0.5, linecolor='black')
# plt.show()
# #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

# 1. 解决 PyCharm 报错：强制使用 TkAgg 后端 (放在所有 plt 调用之前)
matplotlib.use('TkAgg')


# 2. 解决中文乱码：自动寻找系统中的中文字体
def set_chinese_font():
    # 常见中文字体列表
    fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    for font in fonts:
        if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    return None


used_font = set_chinese_font()

# 3. 读取数据
file_name = "pareto_results.xlsx"
if not os.path.exists(file_name):
    print(f"错误：找不到文件 {file_name}，请先运行优化程序。")
else:
    # 读取数据，假设 'Rank' 是索引列
    data = pd.read_excel(file_name, index_col='Rank')

    # 定义列名（根据 帕累托lyk.py 的输出）
    inputs = ['Z1', 'Z2', 'Z3', 'Z5', 'Z6', 'phi_d1', 'phi_d2']
    outputs = ['f1_Loss', 'f2_Weight_N']

    # --- 绘图 1：输入变量之间的相关性 ---
    plt.figure(figsize=(10, 8))
    corr_in_in = data[inputs].corr(method='spearman')  # 推荐用 spearman 捕捉非线性关系
    sns.heatmap(corr_in_in, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=0.5)
    plt.title('设计变量（输入）之间的相关性热图', fontsize=14)
    plt.tight_layout()
    plt.savefig('input_correlation.png', dpi=300)
    print("已保存：设计变量间相关性热图 (input_correlation.png)")

    # --- 绘图 2：输入变量与输出目标的相关性 ---
    plt.figure(figsize=(12, 8))
    # 计算所有列的相关性，但只截取 输入 vs 输出 的部分
    corr_full = data[inputs + outputs].corr(method='spearman')
    corr_in_out = corr_full.loc[inputs, outputs]

    sns.heatmap(corr_in_out, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('设计变量与优化目标的相关性分析', fontsize=14)
    plt.ylabel('设计变量')
    plt.xlabel('优化目标')
    plt.tight_layout()
    plt.savefig('input_output_correlation.png', dpi=300)
    print("已保存：输入与输出相关性热图 (input_output_correlation.png)")

    plt.show()  # 如果这里还是报错，可以注释掉此行，直接去查看保存的 .png 文件