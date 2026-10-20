from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import Dict, List, Any, Union
import uuid
from datetime import datetime
import os
import glob
import sys
import numpy as np
import torch.multiprocessing as mp

from datatools.slice import Slice
from datatools.global_index import get_global_index
from datatools.block_manager import BlockManager
from datatools.reducer.registry import list_reducers

from view_manager import GlobalViewManager
from config import DATASET_DIR,coreset_methods, training_models, fit_mode, extraction_methods
import config

app = Flask(__name__)
CORS(app)

# 静态文件服务配置


@app.route('/static/datasets/<path:filename>')
def serve_dataset_files(filename):
    """提供 datasets 目录下的静态文件服务"""
    try:
        # 使用绝对路径避免工作目录问题
        dataset_dir = DATASET_DIR

        full_path = os.path.join(dataset_dir, filename)

        if not os.path.exists(full_path):
            print(f"文件不存在: {full_path}")
            return jsonify({"error": "File not found"}), 404

        return send_from_directory(dataset_dir, filename)
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "File not found"}), 404


# 初始化全局视图管理器
global_view_manager = GlobalViewManager()

# 动态slice列表
slice_list = Slice.list_all()

# 全局复制粘贴状态
copied_lasso_shape = None
copied_points = False

# 通用响应格式


def make_response(code: int = 0, msg: str = "success", data: Union[Dict, List, None] = None) -> Dict:
    response = {
        "code": code,
        "msg": msg
    }
    if data is not None:
        response["data"] = data
    return response

# 1. 切片管理


@app.route("/api/slices", methods=["GET"])
def get_slices():
    """获取所有切片列表"""
    global slice_list
    # 实时获取最新的slice列表
    get_global_index().reload()
    slice_list = Slice.list_all()
    # print(f"当前切片列表: {slice_list}")
    return jsonify(make_response(data=slice_list))


@app.route("/api/slices/info", methods=["GET"])
def get_slice_info():
    """获取特定切片的详细信息"""
    slice_name = request.args.get("name")
    if not slice_name:
        return jsonify(make_response(1, "Missing slice name")), 400

    # 使用Slice类检查切片是否存在
    try:
        available_slices = Slice.list_all()
        if slice_name not in available_slices:
            return jsonify(make_response(1, "Slice not found")), 404
    except Exception as e:
        return jsonify(make_response(1, f"Error checking slice: {str(e)}")), 500

    try:
        # 使用Slice类获取切片信息
        slice_obj = Slice(slice_name)

        info = {
            "name": slice_name,
            "length": len(slice_obj),
            "global_ids_count": len(slice_obj.global_ids) if hasattr(slice_obj, 'global_ids') else 0,
        }

        return jsonify(make_response(data=info))
    except Exception as e:
        return jsonify(make_response(1, f"Error reading slice: {str(e)}")), 500


@app.route("/api/slices/refresh", methods=["POST"])
def refresh_slices():
    """刷新切片列表（扫描文件夹）"""
    try:
        global slice_list
        slice_list = Slice.list_all()

        return jsonify(make_response(
            msg=f"Slice list refreshed, found {len(slice_list)} slices",
            data=slice_list
        ))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500

# 2. 方法列表获取


@app.route("/api/coreset_methods", methods=["GET"])
def get_coreset_methods():
    """获取核心集选择方法列表"""
    return jsonify(make_response(data=coreset_methods))


@app.route("/api/training_models", methods=["GET"])
def get_training_models():
    """获取训练模型列表"""
    return jsonify(make_response(data=training_models))


@app.route("/api/reduction_methods", methods=["GET"])
def get_reduction_methods():
    """获取降维方法列表"""
    return jsonify(make_response(data=list_reducers()))


@app.route("/api/extraction_methods", methods=["GET"])
def get_extraction_methods():
    """获取特征提取方法列表"""
    return jsonify(make_response(data=extraction_methods))


@app.route("/api/reduction_mode", methods=["GET"])
def get_reduction_mode():
    """获取降维模式列表"""
    return jsonify(make_response(data=fit_mode))


# 3. 核心集选择
@app.route("/api/select_coreset", methods=["POST"])
def select_coreset():
    """应用核心集选择"""
    global slice_list
    try:
        return jsonify(make_response(1, "Coreset selection is not implemented in this research UI; use the bundled DeepCore methods offline.")), 501
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


# 5. 训练与结果
@app.route("/api/train", methods=["POST"])
def train():
    """启动模型训练"""
    try:
        from datatools.selection.coreset import train_from_slice
        data = request.get_json()
        model = data.get("model")
        seed = data.get("seed", 42)
        train_slice = data.get("trainSlice")
        eval_slice = data.get("evalSlice")

        # 验证参数
        if not all([model, train_slice, eval_slice]):
            return jsonify(make_response(1, "Missing required parameters")), 400

        if train_slice not in slice_list or eval_slice not in slice_list:
            return jsonify(make_response(1, "Train or eval slice not found")), 400


        p = mp.Process(
            target=train_from_slice,
            args=(train_slice, eval_slice, model, seed)
        )
        p.start()

        return jsonify(make_response(
            msg="Training started",
            data={"train_id": 1}
        ))
    except Exception as e:
        print(f"Error occurred while starting training: {e}")
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/train_results", methods=["GET"])
def get_train_results():
    """获取训练结果历史"""
    from training_results import parse_results
    model_results=parse_results()
    return jsonify(make_response(data=model_results))

# 6. 可视化数据


@app.route("/api/heatmap", methods=["GET"])
def get_heatmap():
    """获取热力图数据"""
    view_id = request.args.get("view", type=int)
    heat_type = request.args.get("type")

    if view_id is None:
        return jsonify(make_response(1, "Missing view parameter")), 400

    heatmap_data = global_view_manager.get_view_heatmap_data(view_id, heat_type)
    if heatmap_data is None:
        return jsonify(make_response(1, "View not found")), 404

    return jsonify(make_response(data=heatmap_data))


@app.route("/api/scatter", methods=["GET"])
def get_scatter():
    """获取散点数据"""
    view_id = request.args.get("view", type=int)
    x1 = request.args.get("x1", type=float, default=0.0)
    x2 = request.args.get("x2", type=float, default=1.0)
    y1 = request.args.get("y1", type=float, default=0.0)
    y2 = request.args.get("y2", type=float, default=1.0)

    if view_id is None:
        return jsonify(make_response(1, "Missing view parameter")), 400

    # 构建视口边界
    viewport_bounds = {
        "x_min": x1, "x_max": x2,
        "y_min": y1, "y_max": y2
    }

    scatter_data = global_view_manager.get_view_scatter_data(
        view_id, viewport_bounds)
    if scatter_data is None:
        return jsonify(make_response(1, "View not found")), 404

    return jsonify(make_response(data=scatter_data))


@app.route("/api/lasso_sample_images", methods=["GET"])
def get_lasso_sample_images():
    """获取Lasso模式下的采样图片"""
    view_id = request.args.get("view", type=int)

    if view_id is None:
        return jsonify(make_response(1, "Missing view parameter")), 400

    sample_images = global_view_manager.get_view_sample_images(view_id)
    if sample_images is None:
        return jsonify(make_response(1, "View not found")), 404

    return jsonify(make_response(data=sample_images))


@app.route("/api/info", methods=["GET"])
def get_info():
    """获取视图信息"""
    view_id = request.args.get("view", type=int)

    if view_id is None:
        return jsonify(make_response(1, "Missing view parameter")), 400

    info = global_view_manager.get_view_info(view_id)
    if info is None:
        return jsonify(make_response(1, "View not found")), 404

    return jsonify(make_response(data=info))

# 7. 交互


@app.route("/api/lasso_select_points", methods=["POST"])
def lasso_select_points():
    """Lasso选取点"""
    try:
        data = request.get_json()
        view_id = data.get("view")
        lasso_shape = data.get("lassoShape")

        if not view_id:
            return jsonify(make_response(1, "Missing view or lassoShape")), 400


        if len(lasso_shape) > 20:
            # 对lasso_shape进行采样,保留20个点
            sampled_indices = np.linspace(0, len(lasso_shape)-1, 20, dtype=int)
            lasso_shape = [lasso_shape[i] for i in sampled_indices]

        if not lasso_shape or not isinstance(lasso_shape, list) or len(lasso_shape) == 0:
            lasso_shape = None
        # 使用新的视图管理器进行Lasso选择
        selected_points = global_view_manager.lasso_select_in_view(
            view_id, lasso_shape)
        if selected_points is None:
            return jsonify(make_response(1, "View not found")), 404

        return jsonify(make_response(data={"lassoSelection": selected_points}))
    except Exception as e:
        print(f"Error in lasso_select_points: {e}")
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/copy_lasso", methods=["POST"])
def copy_lasso():
    """复制Lasso形状"""
    try:
        data = request.get_json()
        view_id = data.get("view")

        if not view_id:
            return jsonify(make_response(1, "Missing view")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.copy_lasso():
            return jsonify(make_response(1, "No lasso shape to copy")), 400

        return jsonify(make_response(msg="Lasso shape copied"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/copy_points", methods=["POST"])
def copy_points():
    """复制点"""
    try:
        data = request.get_json()
        view_id = data.get("view")

        if not view_id:
            return jsonify(make_response(1, "Missing view")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.copy_points():
            return jsonify(make_response(1, "No points to copy")), 400

        return jsonify(make_response(msg="Points copied"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/paste_points", methods=["POST"])
def paste_points():
    """粘贴点"""
    try:
        data = request.get_json()
        view_id = data.get("view")

        if not view_id:
            return jsonify(make_response(1, "Missing view")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.paste_points():
            return jsonify(make_response(1, "No points to paste")), 400

        return jsonify(make_response(msg="Points pasted"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/delete_lasso_points", methods=["POST"])
def delete_lasso_points():
    """删除Lasso选中的点"""
    try:
        data = request.get_json()
        view_id = data.get("view")

        if not view_id:
            return jsonify(make_response(1, "Missing view")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.delete_lasso_points():
            return jsonify(make_response(1, "No lasso selection to delete")), 400

        return jsonify(make_response(msg="Lasso selected points deleted"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/upsample_lasso_points", methods=["POST"])
def upsample_lasso_points():
    """上采样Lasso选中的点"""
    try:
        data = request.get_json()
        view_id = data.get("view")

        if not view_id:
            return jsonify(make_response(1, "Missing view")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.upsample_lasso_points():
            return jsonify(make_response(1, "No lasso selection to upsample")), 400

        return jsonify(make_response(msg="Lasso selected points upsampled"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/downsample_lasso_points", methods=["POST"])
def downsample_lasso_points():
    """下采样Lasso选中的点"""
    try:
        data = request.get_json()
        view_id = data.get("view")

        if not view_id:
            return jsonify(make_response(1, "Missing view")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.downsample_lasso_points():
            return jsonify(make_response(1, "No lasso selection to downsample")), 400

        return jsonify(make_response(msg="Lasso selected points downsampled"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500

# 8. 切片操作
@app.route("/api/cut_slice", methods=["POST"])
def cut_slice():
    """裁剪切片"""
    try:
        data = request.get_json()
        view_id = data.get("view")
        fraction = data.get("fraction")

        if not view_id or fraction is None:
            return jsonify(make_response(1, "Missing view or fraction")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.cut_slice(fraction):
            return jsonify(make_response(1, "Failed to cut slice")), 500

        return jsonify(make_response(msg=f"Slice cut with fraction {fraction}"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/save_slice", methods=["POST"])
def save_slice():
    """保存切片"""
    try:
        data = request.get_json()
        view_id = data.get("view")
        slice_name = data.get("sliceName")

        if not view_id or not slice_name:
            return jsonify(make_response(1, "Missing view or sliceName")), 400

        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        if not view_manager.save_slice(slice_name):
            return jsonify(make_response(1, "Failed to save slice")), 500

        # 更新slice列表
        global slice_list
        slice_list = Slice.list_all()

        return jsonify(make_response(msg=f"Slice saved as '{slice_name}'"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500

# 9. 视图管理

# 设定模式
@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    """设置模式，并清除现有视图"""
    try:
        data = request.get_json()
        mode = data.get("mode")

        if not mode:
            return jsonify(make_response(1, "Missing mode")), 400

        config.MODE = mode
        # 清除所有现有视图
        global_view_manager.clear_all_views()
        return jsonify(make_response(msg=f"Mode set to {mode}, all views cleared"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/add_view", methods=["POST"])
def add_view():
    """添加视图"""
    try:
        # 使用新的视图管理器创建视图
        view_id, view_manager = global_view_manager.create_view(mode=config.MODE)

        view_config = view_manager.get_config()

        result_data = {"viewId": view_id, "viewConfig": view_config}

        return jsonify(make_response(msg="View added", data=result_data))
    except Exception as e:
        print(f"Error adding view: {e}")
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/delete_view", methods=["POST"])
def delete_view():
    """删除视图"""
    try:
        data = request.get_json()
        view_id = data.get("viewId")

        if view_id is None:
            return jsonify(make_response(1, "Missing viewId")), 400

        # 使用新的视图管理器删除视图
        if not global_view_manager.delete_view(view_id):
            return jsonify(make_response(1, "View not found")), 404

        return jsonify(make_response(msg=f"View {view_id} deleted and views renumbered"))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/views", methods=["GET"])
def get_views():
    """获取所有视图"""
    try:
        view_list = global_view_manager.get_all_views()
        return jsonify(make_response(data=view_list))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/view/<int:view_id>", methods=["GET"])
def get_view(view_id):
    """获取特定视图"""
    try:
        view_manager = global_view_manager.get_view(view_id)
        if view_manager is None:
            return jsonify(make_response(1, "View not found")), 404

        return jsonify(make_response(data=view_manager.get_config()))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


@app.route("/api/view/<int:view_id>", methods=["PUT"])
def update_view(view_id):
    """更新视图配置"""
    try:
        data = request.get_json()
        if not data:
            return jsonify(make_response(1, "No data provided")), 400

        # 使用新的视图管理器更新配置
        if not global_view_manager.update_view_config(view_id, data):
            return jsonify(make_response(1, "View not found or update failed")), 404

        # 获取更新后的配置
        view_manager = global_view_manager.get_view(view_id)
        updated_config = view_manager.get_config()

        return jsonify(make_response(msg="View updated", data=updated_config))
    except Exception as e:
        return jsonify(make_response(1, str(e))), 500


if __name__ == "__main__":
    # 启动时显示配置和slice信息
    mp.set_start_method("spawn", force=True)
    print("=== 数据可视化系统后端启动 ===")
    print(f"找到 {len(slice_list)} 个slice文件:")
    for i, slice_name in enumerate(slice_list, 1):
        try:
            slice_obj = Slice(slice_name)
            print(f"  {i}. {slice_name} (长度: {len(slice_obj)})")
        except Exception as e:
            print(f"  {i}. {slice_name} (加载失败: {e})")

    print("=== 服务器启动中... ===")
    app.run(debug=False, host=config.BACKEND_HOST, port=config.BACKEND_PORT)
