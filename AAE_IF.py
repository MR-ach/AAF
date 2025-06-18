import numpy as np
import pickle
from pyod.models.iforest import IForest  # 修改为PyOD中的Isolation Forest
from pyod.models.deep_svdd import DeepSVDD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import tensorflow as tf
import random

# 环境设置
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
seed = 150
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ================= 配置参数 =================
# FAULTS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']  # 所有类别
FAULTS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']  # 所有类别
NORMAL_CLASS = 'C0'  # 正常类标识
TRAIN_PATH = './data/train/'  # 训练数据路径
TEST_PATH = './data/test/'  # 测试数据路径
FEATURE_PATH = './results/SAE/'  # 特征文件路径

# ============== 辅助函数：标签转换 ==============
def convert_labels(labels):
    """将可能的多维标签转换为一维整数标签"""
    if labels.ndim > 1:
        return labels.argmax(axis=1) if labels.shape[1] > 1 else labels.flatten()
    return labels

# ============== 数据加载与预处理 ==============
# 加载训练数据（仅使用正常类C0）
with open(f'{FEATURE_PATH}{NORMAL_CLASS}_train.pkl', 'rb') as f:
    train_features = pickle.load(f)
    if not isinstance(train_features, np.ndarray):
        train_features = np.array(train_features)
with open(f'{TRAIN_PATH}{NORMAL_CLASS}_label.pkl', 'rb') as f:
    _, train_labels = pickle.load(f)
# 筛选正常样本
train_labels = convert_labels(train_labels)
normal_mask = (train_labels == 0).flatten()
X_train = train_features[normal_mask]
# 标准化器拟合训练数据
scaler = StandardScaler().fit(X_train)

# 加载并合并所有测试数据
X_test, y_test = [], []
for fault in FAULTS:
    with open(f'{FEATURE_PATH}{fault}_test.pkl', 'rb') as f:
        features = pickle.load(f)
    with open(f'{TEST_PATH}{fault}_label.pkl', 'rb') as f:
        _, labels = pickle.load(f)
    labels = convert_labels(labels)
    binary_labels = np.where(labels == 0, 0, 1).astype(int)
    X_test.append(features)
    y_test.append(binary_labels)
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

# ============== 数据标准化 ==============
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============== 模型训练 ==============
iforest = IForest(contamination=0.1, n_estimators=100, random_state=42)  # 修改为IForest
iforest.fit(X_train_scaled)

# ============== 预测与评估 ==============
test_pred = iforest.predict(X_test_scaled)
test_pred_binary = test_pred  # IForest直接输出0/1标签

print("\n全局分类报告:")
print(classification_report(y_test, test_pred_binary,
                            target_names=['正常', '异常']))

# 生成分类报告字典
report_dict = classification_report(y_test, test_pred_binary,
                                    target_names=['正常', '异常'], output_dict=True)

# 自定义格式化函数，保留四位小数
def format_report(report_dict):
    formatted_report = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            formatted_report[key] = {k: f"{v:.4f}" for k, v in value.items()}
        else:
            formatted_report[key] = f"{value:.4f}"
    return formatted_report

# 获取格式化后的报告
formatted_report = format_report(report_dict)

# 打印格式化后的报告
for key, value in formatted_report.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")

print("全局混淆矩阵:")
print(confusion_matrix(y_test, test_pred_binary))


# 新增混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 添加文本标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(y_test, test_pred_binary,
                      classes=['Normal', 'Anomaly'],
                      title='Confusion Matrix')

# ============== 按类别详细分析 ==============
print("\n按类别统计:")
for fault in FAULTS:
    if fault == NORMAL_CLASS:
        mask = (y_test == 0)
        label_type = '正常'
    else:
        mask = (y_test == 1)
        label_type = '异常'
    acc = (test_pred_binary[mask] == y_test[mask]).mean()
    print(f"{fault} ({label_type}): 准确率 {acc:.2%}")

# ============== 可视化配置 ==============
CLASS_COLORS = {
    'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c',
    'C3': '#d62728', 'C4': '#9467bd', 'C5': '#8c564b',
    'C6': '#e377c2', 'C7': '#7f7f7f', 'C8': '#bcbd22', 'C9': '#17becf'
}

# # ============== 改进的可视化函数 ==============
# def plot_separated_tsne(features, true_labels, pred_labels, class_labels,
#                         perplexity=30, random_state=42, sample_size=1000):
#     # 数据采样
#     if len(features) > sample_size:
#         indices = np.random.choice(len(features), sample_size, replace=False)
#         features = features[indices]
#         true_labels = true_labels[indices]
#         pred_labels = pred_labels[indices]
#         class_labels = class_labels[indices]
#     # t-SNE降维
#     tsne = TSNE(n_components=2, perplexity=perplexity,
#                 random_state=random_state, n_jobs=-1)
#     X_embedded = tsne.fit_transform(features)
#     # 真实类别分布
#     plt.figure(figsize=(8, 6))
#     for cls in FAULTS:
#         mask = (class_labels == cls)
#         plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
#                     color=CLASS_COLORS[cls], label=cls,
#                     alpha=0.8, s=25, edgecolors='w')
#     plt.title("True Class Distribution")
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.legend(title="Class", fontsize=10)
#     plt.show()
#     # 预测正确性分布
#     plt.figure(figsize=(8, 6))
#     correct_mask = (true_labels == pred_labels)
#     plt.scatter(X_embedded[correct_mask, 0], X_embedded[correct_mask, 1],
#                 c='#2ca02c', label=f'Correct ({100 * correct_mask.mean():.1f}%)',
#                 alpha=0.7, s=25)
#     plt.scatter(X_embedded[~correct_mask, 0], X_embedded[~correct_mask, 1],
#                 c='#d62728', label=f'Incorrect ({100 * (1 - correct_mask.mean()):.1f}%)',
#                 alpha=0.9, s=25, marker='x')
#     plt.title("Prediction Correctness")
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     # plt.legend()
#     plt.show()
#     # 模型置信度分布
#     plt.figure(figsize=(8, 6))
#     decision_scores = iforest.decision_function(features)  # 使用IForest的决策函数
#     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=decision_scores,
#                 cmap='coolwarm', alpha=0.8, s=25, edgecolors='w')
#     plt.colorbar(label='Decision Score')
#     plt.title("Model Confidence Distribution")
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.show()
#
# # ============== 执行可视化 ==============
# class_labels = []
# for fault in FAULTS:
#     with open(f'{TEST_PATH}{fault}_label.pkl', 'rb') as f:
#         _, labels = pickle.load(f)
#     labels = convert_labels(labels)
#     class_labels.extend([fault] * len(labels))
# class_labels = np.array(class_labels)
#
# plot_separated_tsne(
#     features=X_test_scaled,
#     true_labels=y_test,
#     pred_labels=test_pred_binary,
#     class_labels=class_labels
# )