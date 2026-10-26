# 变截面声管声场的符号恢复与外推

本仓库复现四类声管的PySR符号恢复、空间外推、采样变化、双变量圆锥关系和声压节点/极值分析。训练标签由无损一维Webster初值问题的DOP853数值积分生成；对照包括MLP、Chebyshev多项式及已知正确声场基函数拟合。研究对象是理想数值基准。

仓库代码独立于原`Acoustic_Technique`目录运行。默认路径、既有Conda环境和Julia环境已在`config.json`中设置为当前服务器配置。原有搜索结果不复制为新结果。

## 一键完整复现

在服务器Linux终端执行：

```bash
cd /home/dengyusong/Acoustic_Technique_Github
bash run.sh
```

也可用配置中的Python 3.8以上解释器执行`reproduce.py`。`run.sh`会读取配置并自动切换到该解释器，兼容本服务器较旧的系统Python。入口依次执行环境检查、生成器校验、预实验、主搜索/基线/精修、采样实验/追加、双变量各阶段及最终认证、后处理和汇总。完整设计包含14份清单、447项调度任务，其中157次为PySR搜索。运行时间取决于搜索提前停止、编译及服务器负载。

默认结果保存在：

```text
/home/dengyusong/Acoustic_Technique_Github/runs/full_v1/
```

先检查任务计划或完成小规模实际链路测试：

```bash
bash run.sh --dry-run
bash run.sh --stage smoke
```

`smoke`实际运行12组生成器校验、1次等截面PySR搜索、1组基线、MLP精修和认证，结果在`runs/smoke_v1`；它不代表整套实验已重跑。

## 路径与环境

`config.json`包含仓库路径、输出路径、两个Python解释器、Julia程序与项目、并发及版本期待值。服务器默认使用：

| 用途 | 默认环境 |
| --- | --- |
| 搜索、训练及结构认证 | `/home/dengyusong/anaconda3/envs/gat/bin/python` |
| 独立检查、后处理和绘图 | `/home/dengyusong/anaconda3/envs/sr/bin/python` |
| Julia及后端 | Julia 1.13.1；SymbolicRegression 1.11.3；仓库内`environment/julia`锁定项目 |

入口自身只需要Linux和Python 3.8以上的标准库，不要求先安装本项目。它为各子进程设置`PYTHONPATH`，并调用配置中的解释器。实验代码也支持`pip install -e . --no-deps`，但一键运行不依赖此安装步骤。

其他机器或克隆路径使用：

```bash
python3 reproduce.py --init-config
# 编辑config.local.json中的两个解释器、Julia路径和版本期待值
python3 reproduce.py --config config.local.json
```

环境重建说明见[环境说明](environment/README.md)。入口以JuliaPkg离线模式使用已准备的锁定项目，不在实验期间自动解析或升级依赖；新机器需先按说明安装。Julia的包、编译和下载缓存使用其标准软件目录。

## 分阶段执行与重新实验

```bash
bash run.sh --stage env
bash run.sh --stage generator
bash run.sh --stage pilot
bash run.sh --stage main
bash run.sh --stage sampling
bash run.sh --stage joint
bash run.sh --stage post
```

同名工作区在配置、源码及清单未改变时可继续；已存在的完整结果通常会跳过。入口使用文件锁防止两个驱动器同时写同一工作区。改变代码、配置或清单，或者需要独立重跑时，使用新名称：

```bash
bash run.sh --run-name full_v2
```

后台运行及中断处理见[复现流程](docs/REPRODUCING.md)。

## 结果入口

| 文件或目录 | 内容 |
| --- | --- |
| `runs/full_v1/docs/reproduction_check/本次复现结果.md` | 从本次真实产物计算的阶段统计 |
| `runs/full_v1/docs/reproduction_check/reproduction_check.json` | 完整性、每次认证与有限误差数量 |
| `runs/full_v1/docs/archive_audit.json` | 数据、检查点、候选、图表等归档检查 |
| `runs/full_v1/experiment_00_generator` | 数值标签校验 |
| `runs/full_v1/experiment_01_recovery` | 主搜索与基线及模型 |
| `runs/full_v1/experiment_02_extrapolation` | 域外误差统计和图 |
| `runs/full_v1/experiment_03_sampling` | 采样实验及单独追加 |
| `runs/full_v1/experiment_04_joint` | 双变量原始、focused、追加及基线修订 |
| `runs/full_v1/experiment_05_locations` | 节点、极值及统一边界规则 |
| `runs/full_v1/logs` | 子任务日志、清单退出码、驱动状态和环境版本 |
| `reference_results` | 原研究的小型对照统计与来源记录；不是新运行结果 |

多线程、时间预算和软件环境会影响搜索轨迹。原研究60/60主族恢复、77/80采样族恢复、双变量2/5和4/5是归档对照，不能作为新随机运行必须命中的数字。声压近似、函数族/系数认证和严格恒等分别统计；原始搜索、追加搜索和稀疏化结果分列。完整性检查通过不等于每次结构恢复成功。

本次整理已通过双环境单元检查、实际小型搜索/训练链路，以及旧结果独立副本的完整后处理检查；没有重新执行全套447项任务。证据与范围见[验证记录](docs/VALIDATION.md)。

## 仓库结构与开发

```text
config.json                 路径、环境和并发
reproduce.py / run.sh        一键入口
src/acoustic_technique/      模型、搜索、基线、认证、统计与检查
experiments/manifests/       14份原设计任务清单
environment/                双环境依赖和原环境快照
docs/                       实验设置、复现与迁移说明
reference_results/          冻结的旧结果摘要
tests/                      科学模型与执行计划检查
.github/workflows/          GitHub Actions单元检查
runs/                       本次生成的数据、模型和日志，Git忽略
```

完整设置见[实验设置](docs/EXPERIMENTS.md)，重构范围见[迁移记录](docs/MIGRATION.md)。贡献约定见[CONTRIBUTING.md](CONTRIBUTING.md)。仓库未代作者指定发布许可证；作者和许可信息由权利人在正式发布时填写。

## 方法文献

- Cranmer M. *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl*. 2023. [arXiv:2305.01582](https://arxiv.org/abs/2305.01582).
- Hairer E, Nørsett S P, Wanner G. *Solving Ordinary Differential Equations I: Nonstiff Problems*. 2nd ed. Springer, 1993. [DOI:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1).
- [SciPy 1.10.1 DOP853文档](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.integrate.DOP853.html)。
