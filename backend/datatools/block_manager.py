import os
import json
from typing import Dict, Any
from config import INDEX_DIR

class BlockManager:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.global_index_path = os.path.join(index_dir, 'global_index.json')
        self.blocks_dir = os.path.join(index_dir, 'blocks')
        os.makedirs(self.blocks_dir, exist_ok=True)
        self.global_index = self._load_global_index()

    def _load_global_index(self):
        if os.path.exists(self.global_index_path):
            with open(self.global_index_path, 'r') as f:
                return json.load(f)
        return []

    def _save_global_index(self):
        with open(self.global_index_path, 'w') as f:
            json.dump(self.global_index, f, indent=2)

    def _get_next_start_id(self) -> int:
        if not self.global_index:
            return 0
        last_entry = self.global_index[-1]
        return last_entry["start_id"] + last_entry.get("length", 0)

    def register_block(self, length: int,discription:str, block_info: Dict[str, Any]):
        start_id = self._get_next_start_id()
        block_filename = f'{start_id:08d}.json'
        block_path = os.path.join(self.blocks_dir, block_filename)

        # 存储 block 信息
        with open(block_path, 'w') as f:
            json.dump(block_info, f, indent=2)

        # 更新全局索引
        self.global_index.append({
            "start_id": start_id,
            "length": length,
            "discription": discription,
            "block_path": f'blocks/{block_filename}'
        })
        self._save_global_index()

        return start_id

    def update_block(self, start_id: int, block_info: Dict[str, Any]):
        block_filename = f'{start_id:08d}.json'
        block_path = os.path.join(self.blocks_dir, block_filename)

        # 更新 block 信息
        with open(block_path, 'w') as f:
            json.dump(block_info, f, indent=2)


if __name__ == "__main__":
    manager = BlockManager()
    manager.migrate_feature_to_features_path_dict()
