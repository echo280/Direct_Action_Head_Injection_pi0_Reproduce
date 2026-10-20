# 验收状态

本地生成日期：2026-10-20。

- 已完成：源码固定版本、独立空间分支、配置、数据/权重下载入口、ID 与官方 PRO Task/Position 评估入口、统一 runs 输出、运行记录、文档。
- 本地验证的命令和结果将在仓库的 `LOCAL_VALIDATION.json` 中记录。
- 尚未验证：依赖安装、NVIDIA/JAX 执行、真实权重加载、MuJoCo 状态重放、训练 checkpoint 恢复、rollout。
- 本地 OOD 检查：官方注册表的四个 OOD suite 共 40 任务、80 个 BDDL/状态路径与固定 Hugging Face 文件清单相符；新增单元测试覆盖新指令、新目标、阶段切换、后端隔离、文件完整性及结果配对。
- 尚未实现：真机流程；仿真 OOD 已有入口但尚未真实 rollout 验证。

远端小测试通过后请保留 `runs/reports/model_smoke.json`、环境报告和完整日志；不要把本文件的未验证状态当成已通过。
