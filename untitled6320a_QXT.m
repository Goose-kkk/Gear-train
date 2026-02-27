clc
clear

%% 参数设置
eta_in = 0.99;
eta_out = 0.99;

%% 生成数据网格
i0 = linspace(0.1, 3, 100);   % 避免0值
i1 = linspace(0.1, 3, 100);
[I0, I1] = meshgrid(i0, i1);

%% 初始化结果矩阵
D21 = nan(size(I0));  % 用NaN处理未定义区域
D12 = nan(size(I0));

%% 区域划分计算
mask = I1 < I0;  % 创建逻辑掩码

% i1 < i0区域计算
D21(mask) = ( (I1(mask) - I0(mask)) .* (1 + I1(mask)*eta_in*eta_out) ) ./ ...
            ( (1 + I1(mask)) .* (-I0(mask) + I1(mask)*(eta_in^2)*(eta_out^2)) );

D12(mask) = - ( (1 + I1(mask)) .* (-I1(mask) + I0(mask)*(eta_in^2)*(eta_out^2)) ) ./ ...
            ( eta_in*eta_out .* (I1(mask) - I0(mask)) .* (I1(mask) + eta_in*eta_out) );

% i1 > i0区域计算
mask = I1 > I0;
D21(mask) = - (eta_in*eta_out .* (I1(mask) - I0(mask)) .* (I1(mask) + eta_in*eta_out)) ./ ...
            ( (I1(mask) + 1) .* (-I1(mask) + I0(mask)*(eta_in^2)*(eta_out^2)) );

D12(mask) = ( (1 + I1(mask)) .* (-I0(mask) + I1(mask)*(eta_in^2)*(eta_out^2)) ) ./ ...
            ( (I1(mask) - I0(mask)) .* (1 + I1(mask)*eta_in*eta_out) );

%% 可视化设置
figure('Color','w','Position',[100 100 1200 500])

% D_eta_21曲面
subplot(121)
surf(I0, I1, D21, 'EdgeColor','none')
title('D_{\eta21} 效率曲面')
xlabel('i_0'), ylabel('i_1'), zlabel('D_{\eta21}')
colormap(jet)
colorbar
view(-30,25)

% D_eta_12曲面
subplot(122)
surf(I0, I1, D12, 'EdgeColor','none')
title('D_{\eta12} 效率曲面')
xlabel('i_0'), ylabel('i_1'), zlabel('D_{\eta12}')
colormap(jet)
colorbar
view(-30,25)

%% 添加共同标注
sgtitle('传动系统效率分布曲面 (eta_{in}=0.98, eta_{out}=0.97)')


