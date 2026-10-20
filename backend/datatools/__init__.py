"""
datatools包 - 数据管理工具

包含切片管理、全局索引等功能
"""

# 导入主要类
try:
    from .slice import Slice
    from .global_index import GlobalIndex

    __all__ = ['Slice', 'GlobalIndex']
except ImportError as e:
    print(f"Warning: Failed to import some datatools modules: {e}")
    __all__ = []
