# 最新增强非线性与噪声实验

本版本使用2026年11月20日批次的增强搜索设计。非线性默认是已经确认的`nested80_long`配置，不再以180秒、规模预热或禁止超越函数嵌套的旧配置运行非线性实验。原Webster实验和线性噪声实验仍保留其原设计。

## 运行命令与结果路径

```bash
cd /home/dengyusong/Acoustic_Technique_Github
bash run.sh --stage nonlinear --run-name nonlinear_final_v1
```

这条命令先检查双Python环境和Julia，再生成参考数据，进行五次无噪声确认。至少3/5同时满足基础指标后，固定配置运行50/40/30 dB各五次、A—D独立效率各五次，最后完成20次配对MLP、离线成本统计及绘图。确认种子100—104，噪声种子0—4，效率种子200—204。

若新随机搜索未达到3/5次基础确认，将保存结果并以非零状态结束，后续噪声和成本不会被标记为已完成。多线程且按时间停止的搜索不保证重复获得论文中的成功次数。固定最终配置的重复实验不包含重新选参；如需重走最初的容量探索流程，使用：

```bash
bash run.sh --stage nonlinear-search --run-name nonlinear_capacity_v1
```

该入口执行`experiments/nonlinear_plan.json`中的增强轮次，按域内验证MSE中位数选配置，用独立于探索的100—104种子确认。它始终关闭规模预热、放宽通用嵌套/子树限制，没有回退到最初的受限非线性配置。基础确认阶段最多10小时；起止时刻保存，恢复同一工作区不重置期限。达标后继续噪声和成本，后续用时不受此失败截止限制。两轮确认复用相同编号时，不能当作十个独立的最终配置重复。

其他入口：

```bash
# 原实验 + 80次线性噪声 + 最新非线性完整流程
bash run.sh
# 只运行补充实验（线性噪声 + 最新非线性）
bash run.sh --stage supplemental --run-name supplemental_v1
# 只运行80次线性噪声及配对MLP/切比雪夫/知情基函数
bash run.sh --stage simple-noise --run-name linear_noise_v1
# 30秒非线性实际链路检查，不代表科学目标成功
bash run.sh --stage nonlinear-smoke
# 只重建已有补充结果的统计和图，不训练；使用原运行名称
bash run.sh --stage supplemental-post --run-name nonlinear_final_v1
```

默认完整输出根目录为`runs/full_v2/`。指定`nonlinear_final_v1`时，补充结果在`runs/nonlinear_final_v1/supplemental_study/`：

| 目录或文件 | 内容 |
|---|---|
| `reference/` | 新生成的Burgers干净数据、隐式/级数一致性检查 |
| `jobs/`、`manifests/` | 每次任务的实际配置、种子和批次清单 |
| `supplemental/confirmation_final/` | 五次固定配置无噪声确认及配对MLP |
| `supplemental/exploration_r*/`、`confirmation_r*/` | 仅容量探索入口产生；各轮单列 |
| `supplemental/noise_performance/` | 50/40/30 dB各五次及配对MLP |
| `supplemental/efficiency/` | A/B线性无噪声/40 dB，C/D非线性无噪声/40 dB，20次独立搜索 |
| `supplemental/simple_noise/` | 四构型40 dB及圆锥/悬链线代表配置50/30 dB |
| `summary/all_runs.csv` | 本次逐方法结果、表达式、误差与来源 |
| `summary/grouped_statistics.csv` | 按阶段、场景、参数、SNR、方法、配置、预算及效率组分开统计 |
| `summary/harmonic_successes.csv` | 第1—10阶分别在5%/10%阈值下的通过次数 |
| `summary/summary.json`、`实验结果总结.md` | 本次实际结果和不完整任务索引 |
| `summary/*.png` | 同数据MLP对比、噪声与逐阶谐波、中位表现波形、独立成本 |
| `provenance/`、`status.json`、`controller_finished.json` | 冻结协议、时限、资源与完成状态 |

每个任务保留`job.json`、`configuration.json`、`data.npz`、`candidates.json`、`selected_before_ood.json`、`result.json`、`predictions.npz`、完整PySR检查点和原生日志。非线性另存`harmonics.npz`、`waveforms.npz`、`enhanced_metrics.json`；效率任务另存`cost_selection_before_test.json`与`cost_summary.json`。

## 物理模型、数据与验收

参考场为激波形成前、无耗散Burgers隐式解`P=sin(theta+sigma*P)`，`0≤sigma≤0.8`。使用Brent根求解生成标签，以400项Fubini级数独立交叉核验。训练40位置×64相位、验证24位置×64错位相位均位于`0≤sigma≤0.32`；域内独立测试12×256，域外测试96×256，外推至0.8。参考导数和谐波只用于数值检查与冻结后的评价。

PySR直接接收`(sigma,theta)`和原始声压。候选浮点常数只用训练集精修；在域内验证MSE最低值1.1倍（下限1e-16）窗口内选最简单表达式，先保存`selected_before_ood.json`，再读取测试误差。没有谐波预分离、目标表达式初始化、解析模板、物理残差损失或隐式求解器候选算子。

基础判据同时要求：独立域内声压NRMSE≤1%、域外≤5%、基频和二次谐波各自的相位敏感系数误差≤5%；五次确认至少三次同时达标。逐阶误差按该阶真实谐波能量独立归一化，同时计入正弦系数偏差和余弦系数，包含幅值与相位信息。进阶结果报告第3—10阶的5%及10%通过次数。旧域外1%精度仍另列，不把不同标准混合。

含噪声任务以干净训练声压RMS乘`10^(-SNR/20)`确定高斯噪声标准差，训练与验证使用独立随机数流；配对方法逐点使用同一观测，测试始终干净。MLP仍采用原加强训练流程，但未与PySR匹配计算预算。

## 最终搜索参数与可调整位置

`experiments/nonlinear_plan.json/final_configuration`为最终配置的唯一参数源；`config.json/supplemental`保存并发、数据种子、日志间隔、基础阶段时限和预检预算。路径及Python/Julia环境仍统一放在根配置中。

| 参数 | 最终值 |
|---|---:|
| `timeout_in_seconds` | 3600 |
| `maxsize`、`maxdepth` | 80、30 |
| `populations`、`population_size` | 64、128 |
| `ncycles_per_iteration`、`niterations` | 150、100000000 |
| `warmup_maxsize_by` | 0 |
| `constraints`、`nested_constraints` | 空字典、空字典 |
| `parsimony` | 1e-8 |
| `optimizer_iterations`、`optimizer_nrestarts` | 100、8 |
| `optimize_probability` | 0.3 |
| `batching`、`batch_size` | false、512（完整数据模式下不启用批采样） |
| `precision`、Julia线程 | 64、4 |
| 算子 | `+,-,*,/,sin,cos,exp,cosh` |

取消的是旧函数嵌套和子树大小硬约束及规模预热；表达式规模80和深度30仍用于控制有限搜索。搜索迭代上限设得很高，主要按时间预算结束；编译预热独立计时，与规模预热不是同一概念。各任务的`configuration.json`保存最终生效参数。

## 成本统计与中断

效率实验每个种子一波A—D四条件，最多并行四个4线程搜索。可用CPU少于16时会降低并发并保存实际CPU绑定，不能称为严格复现原服务器的四条件并行资源。线性A/B仍以1%域外精度评价，非线性C/D采用本轮联合基础目标；另外统一报告1%域外快照首次通过时间。

原始日志约20秒记录一次，时间起点为独立Julia编译预热后的日志建立。每个快照先按域内验证选择，再离线检查域外和谐波；首次通过时间是回溯统计，不是实际可用的正确提前停止规则。未通过者按预算截尾留空；最终训练集精修后所选模型与曾通过的原始快照可能不同。

控制器仅停止自己启动的搜索进程组，不停止其他项目。工作区代码/配置/协议指纹发生变化时必须使用新`--run-name`。仓库入口不执行关机操作。

## 代码职责

新代码位于`src/acoustic_technique/supplemental/`，全部使用包内导入，不依赖原`Acoustic_Technique`目录。

| 文件 | 职责 |
|---|---|
| `pipeline.py` | 任务、CPU绑定、确认门控、可选容量探索、噪声、独立成本、MLP及汇总调度 |
| `controller.py` | 同一流程的兼容模块入口 |
| `common.py` | 当前仓库配置、运行路径、表达式求值、训练集常数精修、域内验证选择 |
| `data.py` | 隐式参考、Fubini核验、数据划分、噪声及保存 |
| `search.py` | 默认增强PySR、编译预热、原始搜索、候选精修、冻结后测试 |
| `logger_spec.py`、`search_logger.jl` | 不改动已安装库的原生Hall of Fame快照接口 |
| `evaluation.py`、`enhanced_metrics.py` | 声压、逐阶谐波、导数/物理残差和错位网格检查 |
| `baselines.py` | 同数据MLP、线性切比雪夫与知情基函数 |
| `analyze_cost.py` | 冻结原始快照后离线计算首次通过时刻与截尾 |
| `summarize.py`、`plot.py` | 仅读取本次结果统计、绘图，不导入历史模型作为新重复 |

`reference_results/enhanced/`仅是原增强实验的小型统计及核验参照；不参与训练、候选初始化或本次成功率计算。
