import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import re
from collections import defaultdict


class PreciseConstraintRenderer:
    def __init__(self):
        self.colors = {'W': '#C0392B', 'N': '#2980B9', 'axis': '#7F8C8D', 'const': '#34495E'}
        self.y_s = 0.0
        self.y_p = 2.0
        self.h_half = 1.0  # 使 SW(0+1) 与 PW(2-1) 在 1.0 重合
        self.unit_x = 8.0

        # 用于存储W特征的位置信息
        self.w_positions = {}  # key: 标识符，value: (x位置, y中心, 类型)
        # 用于存储SW元素的位置
        self.sw_positions = []  # 存储所有SW元素的x位置
        # 用于存储行星轮刚性轴端点
        self.planet_axes_endpoints = {}  # key: 行星轮标签，value: (起点, 终点)
        # 用于按顺序记录行星轮出现的位置
        self.planet_occurrences = defaultdict(list)  # key: 行星轮标签，value: [(索引, x位置)]

    def _draw_S_base(self, ax, x):
        """太阳轮基座：在太阳轮中心轴两侧对称添加U形特征，内部完全填充45度倾斜的约束线"""
        # 原有的基座线
        ax.plot([x - 2, x + 2], [self.y_s, self.y_s], color=self.colors['axis'], lw=1.5)

        # 在太阳轮中心轴下方1.0处添加U形开口朝上的特征（左侧）
        u_offset = 1  # 距离中心轴的偏移量
        u_height = 0.6  # U形高度（开口朝上，所以高度向上延伸）
        u_width = 0.5  # U形半宽度

        # U形底部位置（中心轴下方1.0处）
        u_bottom_y = self.y_s - u_offset

        # U形顶部位置（从底部向上延伸u_height）
        u_top_y = u_bottom_y + u_height

        # 左侧U形开口朝上的路径坐标点
        # 这是一个开口朝上的U形，所以从左侧底部开始，向上到顶部，向右移动，再向下到右侧底部
        u_left_points_x = [
            x - u_width - 2.5,  # 左端点（底部）
            x - u_width - 2.5,  # 左上角
            x + u_width - 2.5,  # 右上角
            x + u_width - 2.5  # 右端点（底部）
        ]
        u_left_points_y = [
            u_bottom_y,  # 左端点高度（底部）
            u_top_y,  # 左上角高度（顶部）
            u_top_y,  # 右上角高度（顶部）
            u_bottom_y  # 右端点高度（底部）
        ]

        # 绘制左侧U形路径
        ax.plot(u_left_points_x, u_left_points_y, color=self.colors['const'], lw=1.5)

        # 在左侧U形内部完全填充倾斜45度的约束线
        # 计算U形的宽度和高度
        u_left_width = u_left_points_x[3] - u_left_points_x[0]  # U形宽度
        u_left_height = u_top_y - u_bottom_y  # U形高度

        # 设置斜线间距
        line_spacing = 0.15  # 斜线之间的间距

        # 绘制向右上45度倾斜的斜线，完全填充U形
        # 我们从U形左边界开始，向右移动，直到覆盖整个U形宽度
        current_x = u_left_points_x[0]
        while current_x <= u_left_points_x[3]:
            # 计算当前斜线与U形底部和顶部的交点
            # 斜线方程：y = x + b，其中b是截距

            # 对于给定的x值，斜线与U形底部的交点
            # 由于是45度线，y变化量与x变化量相同
            # 我们从U形底部开始绘制斜线

            # 计算斜线的起始点
            # 我们需要确保斜线在U形内部
            # 方法：从U形底部左侧开始，向右上45度延伸
            start_x = current_x
            start_y = u_bottom_y

            # 计算斜线长度，使其刚好到达U形顶部
            # 斜线在U形内部的长度 = min(到右侧边界的距离, 到顶部的距离)
            # 由于是45度，到顶部的垂直距离 = 顶部y - 起始y
            vertical_distance = u_top_y - start_y

            # 计算在垂直距离内，x可以向右移动的距离
            # 因为45度，x和y变化量相同
            x_distance = vertical_distance

            # 确保不会超出U形右侧边界
            max_x_distance = u_left_points_x[3] - start_x
            actual_x_distance = min(x_distance, max_x_distance)

            # 计算终点坐标
            end_x = start_x + actual_x_distance
            end_y = start_y + actual_x_distance  # 因为45度，y增加量与x相同

            # 确保终点在U形内部
            if end_y <= u_top_y and end_x <= u_left_points_x[3]:
                # 绘制斜线
                ax.plot([start_x, end_x], [start_y, end_y],
                        color=self.colors['const'], lw=0.8)

            # 继续从左侧U形底部右侧开始绘制斜线
            start_x2 = u_left_points_x[0]
            start_y2 = u_bottom_y + (current_x - u_left_points_x[0])

            if start_y2 <= u_top_y:
                # 计算从(start_x2, start_y2)开始的斜线长度
                vertical_distance2 = u_top_y - start_y2
                x_distance2 = min(vertical_distance2, u_left_points_x[3] - start_x2)

                end_x2 = start_x2 + x_distance2
                end_y2 = start_y2 + x_distance2

                if end_y2 <= u_top_y and end_x2 <= u_left_points_x[3]:
                    ax.plot([start_x2, end_x2], [start_y2, end_y2],
                            color=self.colors['const'], lw=0.8)

            current_x += line_spacing

        # 关于y=0对称得到上方U形
        # 对称中心是y = 0，所以对于左侧U形的每个点(x_i, y_i)，对称点为(x_i, -y_i)
        # 因为左侧U形的y坐标都是负值，对称后变成正值，形成开口朝下的U形
        u_upper_points_x = u_left_points_x.copy()  # x坐标保持不变
        u_upper_points_y = [-yi for yi in u_left_points_y]  # y坐标取相反数

        # 绘制上方U形路径
        ax.plot(u_upper_points_x, u_upper_points_y, color=self.colors['const'], lw=1.5)

        # 在上方U形内部完全填充倾斜45度的约束线
        # 上方U形是开口朝下的，顶部在上方U形的最高点
        upper_u_bottom_y = -u_top_y  # 上方U形底部y坐标
        upper_u_top_y = -u_bottom_y  # 上方U形顶部y坐标

        # 绘制向左下45度倾斜的斜线，完全填充上方U形
        current_x = u_upper_points_x[0]
        while current_x <= u_upper_points_x[3]:
            # 计算斜线在U形内部的起始点
            # 我们从U形顶部左侧开始，向左下45度延伸

            # 对于给定的x值，斜线与U形顶部的交点
            start_x = current_x
            start_y = upper_u_top_y

            # 计算斜线长度，使其刚好到达U形底部
            vertical_distance = start_y - upper_u_bottom_y  # 到U形底部的垂直距离

            # 由于是45度，x和y变化量相同
            x_distance = vertical_distance

            # 确保不会超出U形左侧边界（向左下45度，x减小）
            max_x_distance = start_x - u_upper_points_x[0]
            actual_x_distance = min(x_distance, max_x_distance)

            # 计算终点坐标（向左下45度，x减小，y减小）
            end_x = start_x - actual_x_distance
            end_y = start_y - actual_x_distance

            # 确保终点在U形内部
            if end_y >= upper_u_bottom_y and end_x >= u_upper_points_x[0]:
                # 绘制斜线
                ax.plot([start_x, end_x], [start_y, end_y],
                        color=self.colors['const'], lw=0.8)

            # 继续从上方U形顶部右侧开始绘制斜线
            start_x2 = u_upper_points_x[3]
            start_y2 = upper_u_top_y - (current_x - u_upper_points_x[0])

            if start_y2 >= upper_u_bottom_y:
                # 计算从(start_x2, start_y2)开始的斜线长度
                vertical_distance2 = start_y2 - upper_u_bottom_y
                x_distance2 = min(vertical_distance2, start_x2 - u_upper_points_x[0])

                end_x2 = start_x2 - x_distance2
                end_y2 = start_y2 - x_distance2

                if end_y2 >= upper_u_bottom_y and end_x2 >= u_upper_points_x[0]:
                    ax.plot([start_x2, end_x2], [start_y2, end_y2],
                            color=self.colors['const'], lw=0.8)

            current_x += line_spacing

    def _draw_H_feature(self, ax, x, ctype, cid):
        """旋转90度工字(H型)，纵轴中心对齐轴线"""
        cw = self.colors['W']
        y_center = self.y_s if ctype == 'S' else self.y_p

        # 垂直中心轴
        ax.plot([x, x], [y_center - self.h_half, y_center + self.h_half], color=cw, lw=2.5, zorder=5)
        # 上下平行横梁
        ax.plot([x - 1.2, x + 1.2], [y_center + self.h_half, y_center + self.h_half], color=cw, lw=2)
        ax.plot([x - 1.2, x + 1.2], [y_center - self.h_half, y_center - self.h_half], color=cw, lw=2)

        # 如果是太阳轮(S)且是W类型，在H型中心y=0左侧绘制一条长4的横线
        if ctype == 'S':  # 只对S1W添加横线and cid == '1'
            # 在y=0处，H型中心左侧绘制一条长度为4的横线
            # 横线的起点在H型中心左侧，长度为4
            line_start_x = x - 3.5  # 从x-4开始
            line_end_x = x  # 到x结束，长度为4
            line_y = self.y_s  # y=0

            ax.plot([line_start_x, line_end_x], [line_y, line_y],
                    color=cw, lw=2, zorder=5)

            # 记录SW位置
            if ctype == 'S':
                self.sw_positions.append(x)

        # 存储W特征的位置信息
        tag = f"{ctype}{cid}"
        self.w_positions[tag] = (x, y_center, ctype)

        # 文本标注
        y_txt = y_center - 1.5 if ctype == 'S' else y_center + 1.5
        ax.text(x, y_txt, f"{ctype}{cid}", ha='center', fontweight='bold')

    def _draw_C_feature(self, ax, x, cid, planet_axis_x1, planet_axis_x2):
        """绘制系杆C的特征，放在行星轮刚性轴长度中间的x位置"""
        # 计算行星轮刚性轴的中点
        if planet_axis_x1 is not None and planet_axis_x2 is not None:
            # 使用行星轮刚性轴的中点作为系杆C的位置
            c_x = (planet_axis_x1 + planet_axis_x2) / 2
        else:
            # 如果没有行星轮刚性轴信息，使用原来的x位置
            c_x = x

        # 系杆C的中心轴与太阳轮中心轴在同一水平面 (y=0)
        c_y = self.y_s

        # 绘制系杆C的基座U型（与太阳轮形状一致）
        self._draw_S_base(ax, c_x + 2)

        # 从系杆C的中心轴y=0绘制一条1长度的横线（向左）
        line_start_x = c_x - 1.5  # 从c_x-1开始
        line_end_x = c_x  # 到c_x结束，长度为4
        line_y = c_y  # y=0

        ax.plot([line_start_x, line_end_x], [line_y, line_y],
                color=self.colors['const'], lw=2, zorder=5)

        # 转折向上连接到行星轮中心轴下方0.5处
        # 行星轮中心轴在y_p = 2.0处
        turn_point_x = line_start_x  # 转折点在横线左端
        turn_point_y = line_y  # 转折点起始y坐标

        # 垂直向上转折
        vertical_end_y = self.y_p - 0.5  # 行星轮中心轴下方0.5处

        # 绘制转折线（垂直部分）
        ax.plot([turn_point_x, turn_point_x], [turn_point_y, vertical_end_y],
                color=self.colors['const'], lw=1.5, zorder=5)

        # 添加系杆的下横梁（在行星轮中心轴下方0.5处）
        lower_beam_y = self.y_p - 0.5
        beam_length = 2.4  # 横梁长度（上下横梁相同）
        ax.plot([turn_point_x - beam_length / 2, turn_point_x + beam_length / 2],
                [lower_beam_y, lower_beam_y],
                color=self.colors['const'], lw=2, zorder=5)

        # 添加系杆的上横梁（在行星轮中心轴上方0.5处）
        upper_beam_y = self.y_p + 0.5
        # 使用相同的长度
        ax.plot([turn_point_x - beam_length / 2, turn_point_x + beam_length / 2],
                [upper_beam_y, upper_beam_y],
                color=self.colors['const'], lw=2, zorder=5)

        # 添加C标记
        ax.text(c_x - 1.5, c_y - 0.6, f"C{cid}", ha='center', weight='bold', color=self.colors['const'])

    def _draw_beam_at_point(self, ax, x, y, color):
        """在指定位置绘制横梁"""
        ax.plot([x - 1.2, x + 1.2], [y, y], color=color, lw=2, zorder=6)

    def _draw_N_U_refined(self, ax, x, cid, prev_w_tag):
        """
        内啮合(N)U型路径：行星轮轴线 -> 向下 -> 水平 -> 向上 -> 目标点
        并绘制关于行星轮刚性轴对称的第二段U型路径

        prev_w_tag: 前一个W的标识符，例如 "S1" 或 "P1"
        """
        cn = self.colors['N']

        # 获取行星轮刚性轴的端点
        planet_tag = f"P{cid}"
        if planet_tag in self.planet_axes_endpoints:
            start_point, end_point = self.planet_axes_endpoints[planet_tag]
            # 使用行星轮刚性轴的起点作为N路径的起始点
            x_start = start_point
        else:
            # 如果没有行星轮刚性轴端点信息，使用原来的x位置
            x_start = x

        # 获取前一个W的位置信息
        if prev_w_tag in self.w_positions:
            prev_x, prev_y_center, prev_ctype = self.w_positions[prev_w_tag]
            # 下横梁中心位置
            target_y = prev_y_center - self.h_half
            target_x = prev_x
        else:
            # 如果找不到前一个W，使用默认值
            target_y = 1.0
            target_x = x

        # 路径坐标点序列：行星轮轴线端点 -> 向下 -> 水平 -> 向上 -> 目标点
        down_step = 4  # 向下走的距离

        # 第一段U型路径
        path1_x = [x_start, x_start, target_x, target_x, target_x]
        path1_y = [self.y_p, self.y_p - down_step, self.y_p - down_step, target_y, target_y]

        ax.plot(path1_x, path1_y, color=cn, lw=2.5, zorder=4)

        # 在第一段U型路径的终点绘制横梁
        self._draw_beam_at_point(ax, target_x, target_y, cn)

        # 计算关于行星轮刚性轴(y = y_p)对称的第二段U型路径
        # 对于每个点(x_i, y_i)，关于水平线y = y_p对称的点为(x_i, 2*y_p - y_i)
        path2_x = path1_x.copy()
        path2_y = [2 * self.y_p - yi for yi in path1_y]

        # 将第二段U型路径改为实线
        ax.plot(path2_x, path2_y, color=cn, lw=2.5, zorder=4)

        # 在第二段U型路径的终点绘制横梁
        sym_target_y = 2 * self.y_p - target_y
        self._draw_beam_at_point(ax, target_x, sym_target_y, cn)

        # 在水平段中间标注
        mid_x = (x_start + target_x) / 2
        ax.text(mid_x, self.y_p - down_step - 0.2, f"P{cid}-N", color=cn, ha='center', weight='bold')

        # 在对称路径上也添加标注
        ax.text(mid_x, self.y_p + down_step + 0.2, f"P{cid}-N'", color=cn, ha='center', weight='bold')

    def render(self, chain_str):
        tokens = re.findall(r'([SPCW])(\d*)([WN]?)', chain_str)

        # 重置W位置信息和SW位置信息
        self.w_positions = {}
        self.sw_positions = []
        self.planet_axes_endpoints = {}
        self.planet_occurrences = defaultdict(list)

        # 核心逻辑：自动对齐 SW-PW 或 S-PW
        # 新的配对逻辑：对于S1W-P1N-S2W-P1W-C1，P1W要与前面的S2W对齐
        display_groups = []
        processed = [False] * len(tokens)

        # 第一步：找到所有行星轮与太阳轮的配对
        # 对于每个行星轮，找到它前面最近的太阳轮进行配对
        for i in range(len(tokens)):
            if processed[i]:
                continue

            curr_type, curr_id, curr_mtype = tokens[i]

            # 如果是行星轮的W类型，尝试与前面最近的太阳轮W类型配对
            if curr_type == 'P' and curr_mtype == 'W':
                # 向前查找最近的未处理的太阳轮W类型
                found_sun = False
                for j in range(i - 1, -1, -1):
                    if processed[j]:
                        continue
                    prev_type, prev_id, prev_mtype = tokens[j]
                    if prev_type == 'S' and prev_mtype == 'W':
                        # 找到配对的太阳轮
                        display_groups.append([tokens[j], tokens[i]])
                        processed[j] = True
                        processed[i] = True
                        found_sun = True
                        break

                if not found_sun:
                    # 如果没有找到配对的太阳轮，单独显示
                    display_groups.append([tokens[i]])
                    processed[i] = True
            else:
                # 其他情况，暂时不处理，等待后续处理
                continue

        # 第二步：处理剩余未处理的元素
        for i in range(len(tokens)):
            if not processed[i]:
                display_groups.append([tokens[i]])
                processed[i] = True

        # 按照原始顺序对display_groups进行排序
        # 我们需要保持原始的顺序，但确保配对的元素在同一组
        # 重建display_groups以保持原始顺序
        ordered_display_groups = []
        processed = [False] * len(tokens)

        for i in range(len(tokens)):
            if processed[i]:
                continue

            curr_type, curr_id, curr_mtype = tokens[i]

            # 如果是行星轮的W类型，检查是否已经与前面的太阳轮配对
            if curr_type == 'P' and curr_mtype == 'W':
                # 查找这个行星轮是否已经在某个组中
                found_group = False
                for group in display_groups:
                    if len(group) == 2 and group[1] == tokens[i]:
                        # 找到包含这个行星轮的组
                        ordered_display_groups.append(group)
                        # 标记组中的两个元素为已处理
                        for j in range(len(tokens)):
                            if tokens[j] == group[0] or tokens[j] == group[1]:
                                processed[j] = True
                        found_group = True
                        break

                if not found_group:
                    ordered_display_groups.append([tokens[i]])
                    processed[i] = True
            else:
                # 检查这个元素是否已经在某个组中
                found_group = False
                for group in display_groups:
                    if len(group) == 1 and group[0] == tokens[i]:
                        ordered_display_groups.append(group)
                        processed[i] = True
                        found_group = True
                        break
                    elif len(group) == 2 and (group[0] == tokens[i] or group[1] == tokens[i]):
                        # 如果这个元素在某个组中，但已经在前面的行星轮处理过了
                        found_group = True
                        break

                if not found_group:
                    ordered_display_groups.append([tokens[i]])
                    processed[i] = True

        display_groups = ordered_display_groups

        fig, ax = plt.subplots(figsize=(16, 12))  # 增大图形高度
        ax.set_aspect('equal')
        p_track = defaultdict(list)

        # 存储所有token的信息，用于查找前一个W
        token_info = []  # 存储每个token的(类型, id, 啮合类型, x位置)

        # 第一遍：绘制非C元素，并记录行星轮出现的位置和顺序
        for idx, group in enumerate(display_groups):
            x_pos = idx * self.unit_x

            for elem in group:
                ctype, cid, mtype = elem
                tag = f"{ctype}{cid}"
                token_info.append((ctype, cid, mtype, x_pos))

                # 记录行星轮出现的位置（按显示顺序）
                if ctype == 'P':
                    # 记录行星轮的出现顺序和位置
                    # 我们使用一个全局索引来记录顺序
                    occurrence_idx = len(token_info) - 1
                    self.planet_occurrences[tag].append((occurrence_idx, x_pos))

                if ctype == 'S': self._draw_S_base(ax, x_pos)

                if mtype == 'W':
                    self._draw_H_feature(ax, x_pos, ctype, cid)

                if mtype == 'N':
                    # 查找前一个W
                    prev_w_tag = None
                    for i in range(len(token_info) - 2, -1, -1):
                        prev_ctype, prev_cid, prev_mtype, prev_x = token_info[i]
                        if prev_mtype == 'W':
                            prev_w_tag = f"{prev_ctype}{prev_cid}"
                            break

                    # 绘制内啮合U型路径（包含对称路径和终点横梁）
                    self._draw_N_U_refined(ax, x_pos, cid, prev_w_tag)

                    # 记录行星轮出现的位置（按显示顺序）
                    if ctype == 'P':
                        occurrence_idx = len(token_info) - 1
                        self.planet_occurrences[tag].append((occurrence_idx, x_pos))

        # 绘制行星轮刚性轴 - 使用与路径相同的颜色
        # 行星轮刚性轴的起点和终点基于命名链中行星轮首次和末次出现的位置
        cn = self.colors['N']
        for tag, occurrences in self.planet_occurrences.items():
            if len(occurrences) > 0:
                # 按出现顺序排序（按索引）
                sorted_occurrences = sorted(occurrences, key=lambda x: x[0])

                # 第一个出现的x位置作为起点
                first_occurrence = sorted_occurrences[0]
                start_point = first_occurrence[1]

                # 最后一个出现的x位置作为终点
                last_occurrence = sorted_occurrences[-1]
                end_point = last_occurrence[1]

                # 确保起点小于终点
                if start_point > end_point:
                    start_point, end_point = end_point, start_point

                # 存储行星轮刚性轴的端点
                self.planet_axes_endpoints[tag] = (start_point, end_point)

                # 绘制行星轮刚性轴
                ax.plot([start_point, end_point], [self.y_p, self.y_p], color=cn, lw=3)

                # 标记起点和终点（使用英文标注避免字体问题）
                ax.plot([start_point], [self.y_p], marker='o', color='green', markersize=8, zorder=10)
                ax.plot([end_point], [self.y_p], marker='s', color='red', markersize=8, zorder=10)

                # 添加标注（英文）
                ax.text(start_point, self.y_p + 0.3, "Start", ha='center', fontsize=8, color='green')
                ax.text(end_point, self.y_p + 0.3, "End", ha='center', fontsize=8, color='red')

                # 标记所有行星轮出现的位置
                for _, x_pos in sorted_occurrences:
                    ax.plot([x_pos], [self.y_p], marker='x', color='orange', markersize=10, zorder=9)
                    ax.text(x_pos, self.y_p - 0.3, f"{tag}", ha='center', fontsize=8, color='orange')

        # 第二遍：绘制C元素
        for idx, group in enumerate(display_groups):
            x_pos = idx * self.unit_x

            for elem in group:
                ctype, cid, mtype = elem

                if ctype == 'C':
                    # 绘制系杆C特征，放在行星轮刚性轴长度中间的x位置
                    # 假设C1对应P1，根据cid查找对应的行星轮刚性轴
                    planet_tag = f"P{cid}"
                    if planet_tag in self.planet_axes_endpoints:
                        planet_x1, planet_x2 = self.planet_axes_endpoints[planet_tag]
                        self._draw_C_feature(ax, x_pos, cid, planet_x1, planet_x2)
                    else:
                        # 如果没有找到对应的行星轮刚性轴，使用默认位置
                        self._draw_C_feature(ax, x_pos, cid, None, None)

        ax.axis('off')

        # 设置y轴范围以适应所有特征
        ax.set_ylim(-3, 8)  # 扩大y轴范围以显示所有特征

        plt.title(f"Mechanism Topology Inversion: {chain_str}", pad=30, fontsize=14)

        try:
            plt.tight_layout()
            plt.show()
        except:
            plt.savefig("precise_topology_v14.png")
            print("Successfully rendered to 'precise_topology_v14.png'")


if __name__ == "__main__":
    renderer = PreciseConstraintRenderer()
    # 测试链：S1W-P1N-S2W-P1W-C1 (P1W要与前面的S2W对齐)
    renderer.render("S1N-P1W-S1N-P1W-S2W-P1N-S3W-P2N-C1")