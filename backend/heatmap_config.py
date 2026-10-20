"""
热力图配置文件
用于配置不同类型热力图的参数
"""

class HeatmapConfig:
    """热力图配置类"""

    # 通用配置
    DEFAULT_GRID_SIZE = 0.01  # 网格大小（每个网格的边长）

    # KDE配置
    KDE_CONFIG = {
        'n_centers': 100,      # KDE中心点数量
        'sigma': 0.05,         # 高斯核带宽参数
    }

    # LSDD配置
    LSDD_CONFIG = {
        'sigma': 0.01,         # 高斯核带宽参数
        'lambda': 0.01,        # 正则化参数
        'n_iterations': 20,    # 梯度下降迭代次数
    }

    # 类别热力图配置
    CLASS_HEATMAP_CONFIG = {
        'grid_size': 0.01,     # 网格大小
        'normalize': False,    # 是否归一化
        'show_empty_class': -1, # 空网格的类别值
    }

    # 密度差异热力图配置
    DENSITY_DIFF_CONFIG = {
        'grid_size': 0.01,     # 网格大小
        'method': 'lsdd',      # 计算方法：'lsdd' 或 'kde'
        'normalize': True,     # 是否归一化
    }

    # 权重热力图配置
    WEIGHTED_HEATMAP_CONFIG = {
        'grid_size': 0.01,     # 网格大小
        'normalize': True,     # 是否归一化
    }

    @classmethod
    def get_config(cls, heatmap_type: str) -> dict:
        """
        获取指定类型热力图的配置

        Args:
            heatmap_type: 热力图类型 ('class', 'density_diff', 'weighted')

        Returns:
            dict: 配置字典
        """
        config_map = {
            'class': cls.CLASS_HEATMAP_CONFIG,
            'density_diff': cls.DENSITY_DIFF_CONFIG,
            'weighted': cls.WEIGHTED_HEATMAP_CONFIG,
        }

        base_config = {
            'grid_size': cls.DEFAULT_GRID_SIZE,
            'kde': cls.KDE_CONFIG,
            'lsdd': cls.LSDD_CONFIG,
        }

        specific_config = config_map.get(heatmap_type, {})
        base_config.update(specific_config)

        return base_config
