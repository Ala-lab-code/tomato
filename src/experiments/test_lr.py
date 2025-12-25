# test_lr.py
import os
import json
from src.features import create_dataloader
from src.models.baseline_lr import ImprovedLogisticRegressionCV

# 路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints/LR/trained_lr_model.pkl")

# 读取 split_metadata.json
with open(os.path.join(PROCESSED_DATA_DIR, "split_metadata.json"), "r") as f:
    split_metadata = json.load(f)

# 创建测试 DataLoader
test_loader, feature_dim = create_dataloader(
    PROCESSED_DATA_DIR, split='test', img_size=224, batch_size=32,
    split_metadata=split_metadata
)

# 初始化模型并加载权重
model = ImprovedLogisticRegressionCV(num_classes=10, class_weights=split_metadata.get('class_weights'))
model.load_model(MODEL_PATH)

# 在测试集上评估
results = model.evaluate_on_test(test_loader)
print("测试集结果：", results)
