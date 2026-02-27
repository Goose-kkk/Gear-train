% clc;
% clear;
% syms j k
% 
% % Lambda values
% % Lambda_1 = 0.00005;
% % Lambda_2 = 0.00005;
% % Lambda_3 = 0.0021;
% % Lambda_4 = 0.0023;
% % % 
% Lambda_1 = 0.0002;
% Lambda_2 = 0.0002;
% Lambda_3 = 0.0085;
% Lambda_4 = 0.0085;
% % 
% % Lambda_1 = 0.0085;
% % Lambda_2 = 0.0085;
% % Lambda_3 = 0.0085;
% % Lambda_4 = 0.0085;
% 
% % 正向效率
% % 计算分子
% numerator = (((1 - Lambda_1) * Lambda_2 + Lambda_1) * (1 - Lambda_4) + Lambda_3 * (1 - Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4) + Lambda_4) * (1 - (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - j / k));
% % 计算分母
% denominator = (1 / (1 - j) + Lambda_4 + Lambda_2 * (1 - Lambda_4) + (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4));
% % 计算分数
% fraction = numerator / denominator;
% eta_21 = 1 - abs(fraction + (Lambda_1 + Lambda_3 * (1 - Lambda_1)) * (1 - j / k));
% 
% % 反向效率
% % 计算中间表达式
% term1 = (Lambda_2 / (1 - Lambda_2) + Lambda_4) * (1 - j) / (1 - Lambda_4);
% term2 = (Lambda_1 * (1 / Lambda_3 - 1) + 1) * Lambda_3 * ((1 - j / k) * (1 - (Lambda_2 / (1 - Lambda_2) * (1 - j) / (1 - Lambda_4) + Lambda_4 / (1 - Lambda_4) * (1 - j))) + (1 - j) + (Lambda_2 / (1 - Lambda_2) * (1 - j) / (1 - Lambda_4)) + Lambda_4 / (1 - Lambda_4) * (1 - j)) / (1 - Lambda_3 * j / k * (1 + Lambda_1 * (1 / Lambda_3 - 1)));
% % 计算eta12
% eta_12 = 1 - abs(term1 + term2);
% 
% % 将符号表达式转换为函数句柄
% eta_21_func = matlabFunction(eta_21, 'Vars', [j, k]);
% eta_12_func = matlabFunction(eta_12, 'Vars', [j, k]);
% 
% % 创建 j 和 k 的值的网格
% j_values =1:0.01:20;
% k_values =1:0.01:20;
% [J, K] = meshgrid(j_values, k_values);
% 
% % 应用条件 k > j > 1 和 0 < k/j < 1
% mask = K.*J > 1 & J > 1 & K./J > 0 & K./J < 1;
% 
% % 计算 eta_21 和 eta12 的值
% eta_21_values = eta_21_func(J, K);
% eta_12_values = eta_12_func(J, K);
% 
% % 仅在有效区域内绘制
% eta_21_values(~mask) = NaN;
% eta_12_values(~mask) = NaN;
% 
% % 绘制3D图形
% figure;
% surf(J, K, eta_21_values, 'EdgeColor', 'none')
% xlabel('j')
% ylabel('k')
% zlabel('eta_21')
% title('eta_21 vs j and k for k > j > 1 and 0 < k/j < 1')
% 
% colorbar
% 
% figure;
% surf(J, K, eta_12_values, 'EdgeColor', 'none')
% xlabel('j')
% ylabel('k')
% zlabel('eta_12')
% title('eta_12 vs j and k for k > j > 1 and 0 < k/j < 1')
% 
% colorbar
clc;
clear;
syms j k
syms L_1 L_2 L_3 L_4  P_2
% Lambda_1 = 7.258210622411738e-041;
% Lambda_2 = 8.044271180707330e-04;
% Lambda_3 = 0.01;
% Lambda_4 = 0.02;

% Lambda_1 = 0.0002;
% Lambda_2 = 0.0002;
% Lambda_3 = 0.0085;
% Lambda_4 = 0.0085;

% Lambda_1 = 0.0085;
% Lambda_2 = 0.0085;
% Lambda_3 = 0.0085;
% Lambda_4 = 0.0085;

Lambda_1 = 0.01;
Lambda_2 = 0.01;
Lambda_3 = 0.01;
Lambda_4 = 0.01;


% 正向效率
% L_1 =abs(Lambda_1 * ( (1 - j/k)+ (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4) * (1 - (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - j/k)) / (1/( j - 1) + Lambda_4 + Lambda_2 * (1 - Lambda_4) + (Lambda_3 * (1 + Lambda_1) + Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4)) ) )* P_2;
% L_2 = abs((Lambda_2 * (1 - Lambda_4) * (1 - (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - j/k))) / (1/(j -1) + Lambda_4 + Lambda_2 * (1 - Lambda_4) + (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4)) )* P_2;
% L_3 = abs((Lambda_3 * (1 - Lambda_1) / Lambda_1) )* L_1;
% L_4 = abs((Lambda_4 * (1 - (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - j/k))) / (1/(j -1) + Lambda_4 + Lambda_2 * (1 - Lambda_4) + (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4)) )* P_2;
% eta_21 = (P_2 - L_1 - L_2 - L_3 - L_4) / P_2;

% 计算分子
numerator = (((1 - Lambda_1) * Lambda_2 + Lambda_1) * (1 - Lambda_4) + Lambda_3 * (1 - Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4) + Lambda_4) * (1 - (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - j / k));
% 计算分母
denominator = (1 / (j-1) + Lambda_4 + Lambda_2 * (1 - Lambda_4) + (Lambda_3 * (1 - Lambda_1) + Lambda_1) * (1 - Lambda_2 * (1 - Lambda_4) - Lambda_4));
% 计算分数
fraction = numerator / denominator;
eta_21 = 1 - abs(fraction + (Lambda_1 + Lambda_3 * (1 - Lambda_1)) * (1 - j / k));

% 反向效率
% 计算中间表达式
term1 = (Lambda_2 / (1 - Lambda_2) + Lambda_4) * (1 - j) / (1 - Lambda_4);
term2 = (Lambda_1 * (1 / Lambda_3 - 1) + 1) * Lambda_3 * ((1 - j / k) * (1 - (Lambda_2 / (1 - Lambda_2) * (1 - j) / (1 - Lambda_4) + Lambda_4 / (1 - Lambda_4) * (1 - j))) + (1 - j) + (Lambda_2 / (1 - Lambda_2) * (1 - j) / (1 - Lambda_4)) + Lambda_4 / (1 - Lambda_4) * (1 - j)) / (1 - Lambda_3 * j / k * (1 + Lambda_1 * (1 / Lambda_3 - 1)));
% term2 = (Lambda_1 * (1 / Lambda_3 - 1) + 1) * Lambda_3 * ((1 - j / k) * (1 - (Lambda_2 / (1 - Lambda_2) * (j-1) / (1 - Lambda_4) + Lambda_4 / (1 - Lambda_4) * (j - 1))) + (j - 1) + (Lambda_2 / (1 - Lambda_2) * (j - 1) / (1 - Lambda_4)) + Lambda_4 / (1 - Lambda_4) * (j - 1)) / (1 - Lambda_3 * j / k * (1 + Lambda_1 * (1 / Lambda_3 - 1)));
% 计算eta12
eta_12 = 1 - abs(term1 + term2);

% 将符号表达式转换为函数句柄
eta_21_func = matlabFunction(eta_21, 'Vars', [j, k]);
eta12_func = matlabFunction(eta_12, 'Vars', [j, k]);
% 创建 j 和 k 的值的网格
[j_values, k_values] = meshgrid(1:0.01:20, 1:0.01:20);
% 计算 eta_21 和 eta12 的值
eta_21_values = eta_21_func(j_values, k_values);
eta12_values = eta12_func(j_values, k_values);
% 绘制3D图形
figure;
surf(j_values, k_values, eta_21_values, 'EdgeColor', 'none')
xlabel('j')
ylabel('k')
zlabel('eta_21')
title('eta_21 vs j and k for k > j > 1 and 0<k./j<1')
colorbar

figure;
surf(j_values, k_values, eta12_values, 'EdgeColor', 'none')
xlabel('j')
ylabel('k')
zlabel('eta_12')
title('eta_12 vs j and k for k > j > 1and 0<k./j<1')
colorbar