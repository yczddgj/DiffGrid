import os
from datatools.slice import Slice
from datatools.trainer.registry import get_trainer
from config import TRAINING_DIR

def run_trainer(
    task: str,
    train_slice_name: str,
    test_slice_name: str = None,
    seed: int = 42,
    output_dir: str = TRAINING_DIR,
    **trainer_kwargs
):
    """
    训练统一入口，可供 app.py 或 CLI 调用。

    参数:
    - task: "classification" 或 "detection" 等注册任务名
    - train_slice_name: 训练用 Slice 名称
    - test_slice_name: 可选的测试用 Slice 名称
    - seed: 随机种子
    - output_dir: 模型与日志保存路径
    - trainer_kwargs: Trainer 的其他参数，如 batch_size, epochs, model_name 等
    """

    trainer_cls = get_trainer(task)
    if trainer_cls is None:
        raise ValueError(f"未注册的 Trainer 类型: '{task}'")

    print(f"[TrainerRunner] 加载训练 Slice: {train_slice_name}")
    train_slice = Slice(train_slice_name)

    test_slice = None
    if test_slice_name:
        print(f"[TrainerRunner] 加载测试 Slice: {test_slice_name}")
        test_slice = Slice(test_slice_name)

    trainer = trainer_cls(
        train_slice=train_slice,
        test_slice=test_slice,
        seed=seed,
        output_dir=output_dir,
        **trainer_kwargs
    )

    print(f"[TrainerRunner] 启动训练任务: {task}")
    trainer.train()

    if test_slice:
        print("[TrainerRunner] 启动评估...")
        trainer.evaluate()

    print("[TrainerRunner] 保存模型...")
    trainer.save()

    print("[TrainerRunner] 训练完成 ✅")
