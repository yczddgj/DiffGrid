# DiffGrid

论文 **Revealing the Gap: Visual Comparison of Large-Scale Datasets via Multi-Scale Density Difference Map** 的研究代码。

作者：Xinyuan Guo、Xu Zhu、Yilin Ye、Shixia Liu · **CHI 2026**

[论文原文](https://dl.acm.org/doi/full/10.1145/3772318.3791171) · [英文完整说明](README.md)

DiffGrid 将规则化的二维布局与多尺度密度差异图结合，用于交互比较大型数据集的整体与局部分布差异。系统提供散点叠加、样本预览、缩放和平移、套索选择等功能，并保留 KDE 和 LSDD 对比方法。

## 安装与运行

推荐 Python 3.10 / 3.11、Node.js 22 LTS，已在 Linux 上验证。以下命令在仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt
python scripts/prepare_demo.py
```

示例会生成两个各含 300 个样本的合成数据切片及其特征，不需要下载预训练模型。它用于演示系统，与论文中的实验数据分开。

完整的规则化布局需要 Vulkan 1.3 驱动及运行库。安装好系统 Vulkan 环境后：

```bash
python -m pip install -r backend/requirements-vulkan.txt
DIFFGRID_FEATURE_TYPE=demo DIFFGRID_GRID_RESOLUTION=32 python backend/app.py
```

没有 Vulkan 时，可以先显式关闭布局规则化，在 CPU 上体验密度差异图与交互：

```bash
DIFFGRID_FEATURE_TYPE=demo DIFFGRID_REGULARIZATION=0 DIFFGRID_GRID_RESOLUTION=32 python backend/app.py
```

关闭规则化后，空间布局与论文完整方法不同。示例采用较小的 32 × 32 网格，研究代码默认值为 400 × 400。CUDA 版 PyTorch 可自行替换 CPU 安装步骤；CUDA 与 Vulkan 是分别配置的。

在另一个终端启动前端：

```bash
cd frontend
npm ci
npm run dev
```

打开 Vite 输出的本地地址，选择 **DiffGrid** 模式，将 **Demo-current** 设为当前切片，**Demo-reference** 设为参考切片。首次 UMAP 计算需要编译，可能需要稍等片刻。

## 数据和配置

- [数据格式与接入示例](docs/DATA.md)：使用自己的图像、预计算特征与标签。
- [实验脚本与复现范围](docs/EXPERIMENTS.md)：说明现有脚本需要的数据、参数及限制。
- 默认数据目录为仓库下的 `database/`，可以通过 `DIFFGRID_DATA_DIR` 修改；路径不依赖启动时的工作目录。
- 后端默认 `127.0.0.1:5000`。更换端口时，同时设置 `DIFFGRID_PORT` 和前端 `DIFFGRID_PROXY_TARGET`，详见英文 README。
- 原始实验数据、模型权重、参与者记录与本机日志不随代码发布。核心集训练相关工具作为研究工具保留，界面的核心集选择接口尚未实现。

验证命令：

```bash
python -m unittest discover -s tests -v
cd frontend
npm run build
```

引用格式见[英文 README](README.md#citation)和 [CITATION.cff](CITATION.cff)。本项目采用 [MIT 许可证](LICENSE)，第三方代码的许可证和出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
