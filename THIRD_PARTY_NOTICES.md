# Third-party components

The root MIT license applies to the DiffGrid implementation. The following bundled components retain their original notices and licenses.

| Component | Location | Upstream | License |
| --- | --- | --- | --- |
| Scatterplot regularization | `backend/datatools/reducer/de-cluttering/` | [hennesrave/regularization-python](https://github.com/hennesrave/regularization-python) | [MIT, copyright 2024 Hennes Rave](backend/datatools/reducer/de-cluttering/LICENSE) |
| DeepCore and local adaptations | `backend/deepcore/` | [PatrickZH/DeepCore](https://github.com/PatrickZH/DeepCore) | [MIT, copyright 2023 ZHAO, BO](backend/deepcore/LICENSE.md) |

The regularization directory includes upstream compiled SPIR-V shaders. Local Python adapter changes make Vulkan validation layers optional, copy returned arrays before unmapping device memory, and release Vulkan resources explicitly. The algorithm and upstream license are retained. See also the [upstream regularization project](https://github.com/hennesrave/regularization) for its shader implementation.

Some DeepCore modules identify additional upstream implementations in their source headers (including MobileNetV3, WideResNet, and CIFAR model implementations); these headers are preserved. DiffGrid also uses external libraries installed through pip and npm, which carry their own licenses. Pretrained models and datasets are acquired separately and are not licensed by this repository.

Related work:

- Hennes Rave, Vladimir Molchanov, and Lars Linsen. *De-cluttering Scatterplots with Integral Images*. IEEE Transactions on Visualization and Computer Graphics, 2024. [DOI](https://doi.org/10.1109/TVCG.2024.3381453).
- Chengcheng Guo, Bo Zhao, and Yanbing Bai. *DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning*. [Paper](https://arxiv.org/abs/2204.08499), [code](https://github.com/PatrickZH/DeepCore).

The system figure in `docs/images/overview.png` is copied from the authors' [DiffGrid publication page](https://shixialiu.com/#paper_diffgrid) ([original image](https://shixialiu.com/publications/diffgrid/thumbnail.jpg)). The root MIT license applies to code, not to this paper figure.
