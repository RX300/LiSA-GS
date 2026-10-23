# 项目范围

LisaGS 是 PORT-GS 最终 LiSA-staged 的独立项目，研究已知点光源下的逐场景 3DGS 重光照。
SSD-GS 是工作区选定的比较基线，LisaGS 自身不导入 SSD-GS 或 PORT-GS 的 Python 源码。

只保留当前最终实现：原生 3DGS、自身轮廓初始化、光空间图集、SVBRDF 材质头、学习法线、
高光 lobes 和 30k+8k 训练流程。论文使用的当前组件消融开关仍在同一个方法中；
旧 LiSA compact/reflect/spatial 头、directional_port_v1、surface_attention、neural_material、
GGGS、2DGS、SDF 和旧独立残差训练项目均未迁移。

当前任务是工程隔离，不改变方法设计、训练预算或评价协议。
详细内容见 [原理](method/principles.md)和 [运行协议](experiments/setup.md)。
