# 修改与验证

代码采用`src/acoustic_technique`布局，模块使用相对导入。所有科学计算命令在Linux执行。提交前运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python reproduce.py --dry-run
```

涉及搜索、网络训练或运行环境的修改还应使用配置指定的双环境完成`--stage smoke`。完整实验用新的`--run-name`，保留失败结果。改变算子、预算、数据切分、接受规则或证书门槛属于实验设计修改，须同时更新清单、方法说明和结果口径。

格式化使用Black 24.8.0，行宽100。生成数据、模型及日志保存在被Git忽略的`runs/`中。不要把机器私钥、口令、账户令牌或大型训练产物加入提交。原始基准结果与新的复现结果分开保存。
