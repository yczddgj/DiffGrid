# DiffGrid

论文 [**Revealing the Gap: Visual Comparison of Large-Scale Datasets via Multi-Scale Density Difference Map**](https://dl.acm.org/doi/full/10.1145/3772318.3791171) 的代码。

Xinyuan Guo、Xu Zhu、Yilin Ye、Shixia Liu · **CHI 2026** · [English](README.md)

![DiffGrid 系统界面](docs/images/overview.png)

论文系统图，来自[作者的论文主页](https://shixialiu.com/#paper_diffgrid)。

## 安装

需要 Python 3.10/3.11、Node.js 22，以及 Vulkan 1.3 和兼容驱动。已在 Linux 上测试。

```bash
git clone https://github.com/yczddgj/DiffGrid.git
cd DiffGrid
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt -r backend/requirements-vulkan.txt
```

使用 GPU 时，将上面的 CPU 版 PyTorch 换为对应的 CUDA 版本。

## 运行

生成合成示例数据，启动后端：

```bash
python scripts/prepare_demo.py
DIFFGRID_FEATURE_TYPE=demo DIFFGRID_GRID_RESOLUTION=32 python backend/app.py
```

没有 Vulkan 时，在后端命令前加上 `DIFFGRID_REGULARIZATION=0`，使用不含布局规则化的版本。

另开终端启动前端：

```bash
cd frontend
npm ci
npm run dev
```

打开 `http://localhost:5173`，选择 **DiffGrid**，将两个切片设为 **Demo-current** 和 **Demo-reference**，点击 **Apply**。后端默认地址为 `127.0.0.1:5000`。

## 数据与引用

仓库不包含原始数据和模型权重。数据接入见[数据说明](docs/DATA.md)，实验脚本和配置见[实验说明](docs/EXPERIMENTS.md)。默认数据目录为 `database/`，可用 `DIFFGRID_DATA_DIR` 修改。

引用格式见[英文 README](README.md#citation)或 [CITATION.cff](CITATION.cff)。代码采用 [MIT 许可证](LICENSE)，第三方代码及图片出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
