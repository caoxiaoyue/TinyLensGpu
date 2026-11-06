# J0946 实验图的证据约定

问题：固定既有 M3 模型后的迭代校正能否给出通过数值检查的 source/potential 场？
图只展示搜索内最高合格 evidence 对应的实测数据拟合，不证明 subhalo 检测。

采用 Python/matplotlib，与本任务既有绘图工作一致。图型为 image plate，六格依次为
减透镜光后的观测、校正后前向模型、同一 noise 下的标准化残差、M3 原始标准化残差、
恢复的 δκ、恢复 source。前四格负责拟合与基线比较，后两格给出对应场；没有真值面板。

使用保存 NPZ 所有原始数据、固定 mask/noise。显示 bbox 包含全部保留像素与全部
potential 节点；mask 排除区域为浅灰，κ 只在 derivative_valid 节点展示，无外推/插值。
观测/模型共享显示限（保留像素的 0.5/99.5 分位，仅影响颜色）；两个 residual 共享
[-5,5] sigma 色限；source 保留 signed 值；δκ 使用对称色限。超限值只颜色饱和，不从
拟合或统计中删除。source 单独使用 source-plane arcsec 坐标，其他为 image plane。

输出 PNG/PDF/SVG 及量化 source-data NPZ。报告图不是投稿规格；设置字体 ≥8pt、
PDF 字体42/SVG可编辑文本，并在导出前测量所有主绘图区对齐，导出后运行 PDF
文字和碰撞审计及最终图像目视检查。真实数据无 injected truth，无 uncertainty 或
质量显著性推断，conditional 宏模型不确定性没有传播。

最终图显示所选单次合格候选，并在图内标记 cold-start repeat gate NOT PASSED。
后一次独立进程通过不改变这一保守标注。最终绘图区对齐/渲染碰撞0 fail/0 warn，
PDF最小文字8pt；图源预检0 fail。264mm报告幅面、300dpi PNG与未生成投稿TIFF的
3项提醒属于报告导出约定，不作投稿规格达标声明。
