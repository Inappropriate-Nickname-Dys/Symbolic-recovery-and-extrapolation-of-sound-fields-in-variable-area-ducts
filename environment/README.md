# 环境配置

当前服务器已有`gat`和`sr`，根目录`config.json`已指向它们。Julia项目锁定文件位于仓库的`environment/julia/`，运行入口设置`PYTHON_JULIAPKG_OFFLINE=yes`，不在实验期间自动解析或安装依赖。搜索用PySR 1.5.9、Julia 1.13.1及SymbolicRegression 1.11.3；分析环境与搜索环境的NumPy、SciPy、SymPy和Torch版本分别记录，避免把两份依赖混装。

迁移到新机器时，可先创建Python环境：

```bash
cd environment
conda env create -f regression.yml
conda env create -f analysis.yml
cd ..
python3 reproduce.py --init-config
```

随后编辑`config.local.json`中的解释器路径。新环境名称为`acoustic-regression`和`acoustic-analysis`；原服务器的`gat`和`sr`不受影响。

Julia应另行安装与归档相同的版本。`julia/Project.toml`与`julia/Manifest.toml`来自实际归档环境，包含PythonCall等依赖。新机器需在仓库根目录使用该版本的Julia显式准备包（将下方程序路径换为实际路径）：

```bash
/path/to/julia-1.13.1/bin/julia --project=environment/julia -e 'using Pkg; Pkg.instantiate()'
```

随后在本机配置中填写Julia程序路径，`--init-config`会将Julia项目路径设为当前仓库下的`environment/julia`。保留`archived/julia_*`作为来源记录。仅安装两个Python环境不足以完成Julia后端准备。

使用其他版本时应同步调整版本期待值并记录环境差异；这不属于原版本的严格环境复现。检查命令：

```bash
python3 reproduce.py --config config.local.json --stage env
python3 reproduce.py --config config.local.json --stage smoke
```

`requirements-*.txt`是本项目主要依赖的版本固定文件，`archived/python_environment.txt`包含原回归环境的更完整包清单。Julia和Conda自身管理的包、编译、下载缓存位于其正常软件目录；实验产物在仓库`runs/`内。
