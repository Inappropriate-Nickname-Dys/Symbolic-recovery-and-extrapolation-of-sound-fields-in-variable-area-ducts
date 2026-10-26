# 完整运行和故障恢复

## 推荐顺序

首次使用先执行`bash run.sh --dry-run`和`bash run.sh --stage smoke`。确认配置与真实小型链路均可运行后，执行`bash run.sh`。默认完整结果为`runs/full_v1`，smoke为`runs/smoke_v1`；可通过`--run-name`另设名称。

完整入口的顺序为：环境与Julia版本；12组生成器校验；单/双变量预实验；60次主搜索、60组基线、60组精修；80次采样搜索、80组基线、80组精修、1次采样追加；双变量main、原始基线、基线修订、focused、900秒追加；最后认证、独立检查、稀疏化、统计、绘图及完整性核验。所有搜索的原始任务定义在`experiments/manifests`。

主搜索和采样的并发在`execution.workers`中配置，双变量常规批次在`joint_workers`中配置，两个追加批次单独运行。原服务器80个逻辑CPU，默认主并发12、单任务4个Julia线程；900秒双变量追加由清单明确指定16线程。过高并发会改变按时间预算搜索的轨迹。

## 后台执行

```bash
cd /home/dengyusong/Acoustic_Technique_Github
mkdir -p runs
nohup bash run.sh > runs/full_driver.log 2>&1 < /dev/null &
printf '%s\n' "$!" > runs/full_driver.pid
```

查看`tail -n 40 runs/full_driver.log`和`ps -p "$(cat runs/full_driver.pid)" -o pid,etime,args`。进程消失后应检查日志是否出现`REPRODUCTION_COMPLETE all`，不能仅凭退出推断完成。每次步骤的命令、退出码及日志写入运行目录`logs/driver_status_*.json`，调度器还保存每份清单的`*_status.json`及逐label日志。

## 分阶段和继续运行

运行器支持`env/generator/pilot/main/sampling/joint/post`，使用同一配置和运行名称即可继续。原基线必须先于精修，所有训练及生成器必须先于完整post。工作区冻结代码、任务清单和配置的指纹；发生变更时要求新名称，以防一次统计混用多个实验设计。

遇到非零退出码，查看驱动状态定位步骤，再看具体label日志。多数运行器发现最终结果文件后会跳过。若已有结果损坏，或仅有部分PySR文件而没有最终结果，先确认没有活跃进程，在同一工作区`interrupted/`下保留该不完整目录后再运行对应阶段。不要删除完整失败案例，也不要把新结果移入旧归档。

同一运行目录有驱动器文件锁，不允许并发的第二个驱动器。直接手动调用底层模块不经过该锁，因此不要同时使用驱动器和手动任务写同一目录。

## 输出核验

完整post写入`docs/reproduction_check/本次复现结果.md`和同目录JSON，按阶段报告恢复数、严格恒等及误差分布，并检查447项任务产物。另读`docs/archive_audit.json`和`docs/audit.json`，确认归档、训练/验证边界、冻结式与基线数组一致。

每个搜索目录保存配置、数据、候选、原式/精修式、冻结选择、预测、证书及PySR检查点。各基线目录保存原MLP、精修MLP、预测、Chebyshev参数和连续导数诊断。稀疏化只在有相关候选并通过接受规则时产生`pruned_result.json`，它不是每次必需文件。

原始成功率与追加、后处理分别统计。新候选若声压误差小但族认证或初值导数失败，仍记为结构恢复失败。未被保守语法分类的候选需人工检查表达式，不可直接断言理论不可恢复。

## 报告与图

自动摘要读取本次真实JSON，不复用原研究中写死的成功结论。统计图中的固定代表设置仍为1号组种子0；近入口诊断保留双变量main种子2的来源，但使用中性图题，实际是否有异常按本次曲线和独立核验判断。只有旧表达式才具有旧论文图5的特定δ和`t/δ`尺度。

原仓库包含8幅服务器PNG及对应SVG，论文中的5幅图由另外的排版流程重绘。这里不自动把本次新结果写入旧Word。完整实验成功的标准是程序、归档和统计有效，科研恢复率应据实报告。
