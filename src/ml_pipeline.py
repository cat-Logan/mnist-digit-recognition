# -*- coding: utf-8 -*-
"""
《机器学习》课程设计 —— 基于CNN/SVM的手写数字识别系统
ML Pipeline: 数据探索 → 预处理 → CNN训练 → SVM对比 → 评估对比 → 保存模型

Author: 课程设计
Dataset: MNIST (手写数字 0-9)
来源: Yann LeCun's MNIST / torchvision.datasets.MNIST
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, f1_score, precision_score, recall_score)
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

# ======================== 设置 ========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
print(f"PyTorch 版本: {torch.__version__}")

# ======================== 1. 数据加载与探索 ========================
print("=" * 70)
print("第1步：数据加载与探索性分析 (EDA)")
print("=" * 70)

# ---- 从快速镜像下载MNIST (避免yann.lecun.com的慢速) ----
import gzip, urllib.request, shutil

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_RAW_DIR = os.path.join(DATA_DIR, 'mnist_raw', 'MNIST', 'raw')
os.makedirs(MNIST_RAW_DIR, exist_ok=True)

mnist_files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]

for fname in mnist_files:
    fpath = os.path.join(MNIST_RAW_DIR, fname)
    if not os.path.exists(fpath):
        url = MNIST_MIRROR + fname
        print(f"  下载 {fname} ...")
        try:
            urllib.request.urlretrieve(url, fpath)
        except Exception as e:
            print(f"  镜像下载失败，尝试备用源...")
            urllib.request.urlretrieve(
                f"https://github.com/cvdfoundation/mnist/raw/master/data/{fname}", fpath)
        print(f"  {fname} 下载完成")
    else:
        print(f"  {fname} 已存在，跳过下载")

print("  MNIST文件准备完成")

# 用 torchvision 加载（download=False 因为文件已就位）
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(5),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_full = datasets.MNIST(
    root=os.path.join(DATA_DIR, 'mnist_raw'), train=True,
    download=True, transform=transform_train
)
test_dataset = datasets.MNIST(
    root=os.path.join(DATA_DIR, 'mnist_raw'), train=False,
    download=True, transform=transform_test
)

# 划分验证集：从训练集中分出 10% 作为验证集
val_size = int(0.1 * len(train_dataset_full))
train_size = len(train_dataset_full) - val_size
train_dataset, val_dataset = random_split(
    train_dataset_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\n数据集来源: MNIST (Yann LeCun, Corinna Cortes, Christopher J.C. Burges)")
print(f"获取链接: http://yann.lecun.com/exdb/mnist/")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"图像尺寸: 28x28 灰度图")
print(f"类别数: 10 (数字 0-9)")

# 加载原始数据用于可视化和SVM
raw_train = datasets.MNIST(
    root=os.path.join(DATA_DIR, 'mnist_raw'), train=True, download=False,
    transform=transforms.ToTensor()
)
raw_test = datasets.MNIST(
    root=os.path.join(DATA_DIR, 'mnist_raw'), train=False, download=False,
    transform=transforms.ToTensor()
)

X_train_all = raw_train.data.numpy().reshape(-1, 784) / 255.0
y_train_all = raw_train.targets.numpy()
X_test_all = raw_test.data.numpy().reshape(-1, 784) / 255.0
y_test_all = raw_test.targets.numpy()

print(f"\n原始数据形状: X_train={X_train_all.shape}, y_train={y_train_all.shape}")

# ---- EDA 可视化 ----
# 1) 样本展示
fig, axes = plt.subplots(4, 10, figsize=(16, 7))
for i in range(10):
    idx = np.where(y_train_all == i)[0][0]
    axes[0, i].imshow(raw_train.data[idx], cmap='gray')
    axes[0, i].set_title(f'数字{i}', fontsize=9)
    axes[0, i].axis('off')
for i in range(10):
    idx = np.where(y_train_all == i)[0][1]
    axes[1, i].imshow(raw_train.data[idx], cmap='gray')
    axes[1, i].axis('off')
for i in range(10):
    idx = np.where(y_train_all == i)[0][10]
    axes[2, i].imshow(raw_train.data[idx], cmap='gray')
    axes[2, i].axis('off')
for i in range(10):
    idx = np.where(y_train_all == i)[0][50]
    axes[3, i].imshow(raw_train.data[idx], cmap='gray')
    axes[3, i].axis('off')
plt.suptitle('MNIST手写数字样本展示 (每行展示同数字的不同书写风格)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mnist_samples.png'))
plt.close()
print("图1已保存: mnist_samples.png")

# 2) 类别分布
fig, ax = plt.subplots(figsize=(10, 5))
counts = np.bincount(y_train_all)
colors_bar = plt.cm.tab10(np.arange(10))
ax.bar(range(10), counts, color=colors_bar, edgecolor='white')
for i, c in enumerate(counts):
    ax.text(i, c + 50, str(c), ha='center', fontsize=10, fontweight='bold')
ax.set_xlabel('数字类别', fontsize=12)
ax.set_ylabel('样本数量', fontsize=12)
ax.set_title('MNIST训练集各类别样本分布', fontsize=14, fontweight='bold')
ax.set_xticks(range(10))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'))
plt.close()
print("图2已保存: class_distribution.png")

# 3) 像素均值图
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    mean_img = raw_train.data[y_train_all == i].float().mean(dim=0)
    im = ax.imshow(mean_img, cmap='hot')
    ax.set_title(f'数字 {i} 均值图', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('MNIST每类数字的平均像素激活图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mean_digits.png'))
plt.close()
print("图3已保存: mean_digits.png")

# ======================== 2. CNN 模型定义 ========================
print("\n" + "=" * 70)
print("第2步：定义CNN卷积神经网络模型")
print("=" * 70)

class CNN(nn.Module):
    """轻量级CNN：2层卷积 + 2层全连接"""
    def __init__(self, num_classes=10, dropout_rate=0.25):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = CNN(num_classes=10, dropout_rate=0.25).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nCNN 模型结构:")
print(f"  Conv1: 1->32, 3x3, BN, ReLU, Pool(2x2)")
print(f"  Conv2: 32->64, 3x3, BN, ReLU, Pool(2x2)")
print(f"  FC1: 3136->256, ReLU, Dropout")
print(f"  FC2: 256->10, Softmax")
print(f"  总参数量: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")

# ======================== 3. CNN 训练 ========================
print("\n" + "=" * 70)
print("第3步：CNN模型训练")
print("=" * 70)

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0.0
best_model_state = None

for epoch in range(EPOCHS):
    # ---- 训练 ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # ---- 验证 ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = 100.0 * val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc)

    scheduler.step(val_epoch_acc)

    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        best_model_state = model.state_dict().copy()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
          f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}% | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

print(f"\n最佳验证准确率: {best_val_acc:.2f}%")
model.load_state_dict(best_model_state)

# ---- 训练曲线 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, EPOCHS+1), train_losses, 'o-', label='训练损失', color='#FF6B6B', lw=2)
ax1.plot(range(1, EPOCHS+1), val_losses, 's-', label='验证损失', color='#4ECDC4', lw=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('CNN训练 & 验证损失曲线', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS+1), train_accs, 'o-', label='训练准确率', color='#FF6B6B', lw=2)
ax2.plot(range(1, EPOCHS+1), val_accs, 's-', label='验证准确率', color='#4ECDC4', lw=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('CNN训练 & 验证准确率曲线', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cnn_training_curves.png'))
plt.close()
print("图4已保存: cnn_training_curves.png")

# ==================== 4. CNN 测试集评估 ====================
print("\n" + "=" * 70)
print("第4步：CNN测试集评估")
print("=" * 70)

model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

cnn_acc = accuracy_score(all_labels, all_preds)
print(f"\nCNN 测试集准确率: {cnn_acc:.4f}")
print(f"\nCNN 分类报告:")
print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

# 混淆矩阵
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10),
            cbar_kws={'label': '归一化比例'})
ax.set_xlabel('预测类别', fontsize=12)
ax.set_ylabel('真实类别', fontsize=12)
ax.set_title(f'CNN 混淆矩阵 (测试准确率: {cnn_acc:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cnn_confusion_matrix.png'))
plt.close()
print("图5已保存: cnn_confusion_matrix.png")

# 保存CNN模型
cnn_model_path = os.path.join(MODELS_DIR, 'cnn_mnist.pth')
torch.save({
    'model_state_dict': best_model_state,
    'val_acc': best_val_acc,
    'model_config': {
        'num_classes': 10, 'dropout_rate': 0.25
    }
}, cnn_model_path)
print(f"CNN模型已保存: {cnn_model_path}")

# TorchScript trace (try, non-critical)
try:
    dummy_input = torch.randn(1, 1, 28, 28).to(DEVICE)
    traced_model = torch.jit.trace(model.cpu(), dummy_input.cpu())
    traced_path = os.path.join(MODELS_DIR, 'cnn_mnist_traced.pt')
    traced_model.save(traced_path)
    print(f"CNN TorchScript模型已保存: {traced_path}")
except Exception as e:
    print(f"[INFO] TorchScript导出跳过 ({e})，不影响使用")

# ==================== 5. SVM 对比模型 ====================
print("\n" + "=" * 70)
print("第5步：SVM对比模型训练")
print("=" * 70)

# SVM在原始像素上计算量大，使用策略：
# 1. 取训练集子集用于GridSearch
# 2. 使用PCA降维

# 子采样 (分层采样，每类取1000个 = 10000总样本用于训练SVM)
np.random.seed(42)
svm_X, svm_y = [], []
for c in range(10):
    idx_c = np.where(y_train_all == c)[0]
    chosen = np.random.choice(idx_c, size=min(1000, len(idx_c)), replace=False)
    svm_X.append(X_train_all[chosen])
    svm_y.append(y_train_all[chosen])
svm_X = np.vstack(svm_X)
svm_y = np.hstack(svm_y)

print(f"SVM训练样本数: {len(svm_X)} (每类1000个)")

# PCA降维到100维
pca = PCA(n_components=100, random_state=42)
X_svm_pca = pca.fit_transform(svm_X)
X_test_pca = pca.transform(X_test_all)

print(f"PCA: 784维 -> 100维 (保留方差比: {pca.explained_variance_ratio_.sum():.4f})")

# 标准化
scaler_svm = StandardScaler()
X_svm_pca_scaled = scaler_svm.fit_transform(X_svm_pca)
X_test_pca_scaled = scaler_svm.transform(X_test_pca)

# SVM 训练 (简化参数网格以加速)
print("\n训练 SVM (RBF核)...")
svm_model = SVC(
    C=10, kernel='rbf', gamma='scale', probability=True,
    random_state=42, cache_size=2000
)
svm_model.fit(X_svm_pca_scaled, svm_y)

svm_train_acc = accuracy_score(svm_y, svm_model.predict(X_svm_pca_scaled))
svm_test_pred = svm_model.predict(X_test_pca_scaled)
svm_test_acc = accuracy_score(y_test_all, svm_test_pred)
svm_test_prob = svm_model.predict_proba(X_test_pca_scaled)

print(f"SVM 训练准确率: {svm_train_acc:.4f}")
print(f"SVM 测试准确率: {svm_test_acc:.4f}")
print(f"\nSVM 分类报告:")
print(classification_report(y_test_all, svm_test_pred, target_names=[str(i) for i in range(10)]))

# SVM 混淆矩阵
fig, ax = plt.subplots(figsize=(10, 8))
cm_svm = confusion_matrix(y_test_all, svm_test_pred)
cm_svm_norm = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_svm_norm, annot=cm_svm, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=range(10), yticklabels=range(10),
            cbar_kws={'label': '归一化比例'})
ax.set_xlabel('预测类别', fontsize=12)
ax.set_ylabel('真实类别', fontsize=12)
ax.set_title(f'SVM 混淆矩阵 (测试准确率: {svm_test_acc:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'svm_confusion_matrix.png'))
plt.close()
print("图6已保存: svm_confusion_matrix.png")

# 保存SVM相关模型
joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_mnist.pkl'))
joblib.dump(pca, os.path.join(MODELS_DIR, 'pca_mnist.pkl'))
joblib.dump(scaler_svm, os.path.join(MODELS_DIR, 'scaler_svm.pkl'))
print(f"SVM模型已保存: {os.path.join(MODELS_DIR, 'svm_mnist.pkl')}")
print(f"PCA模型已保存: {os.path.join(MODELS_DIR, 'pca_mnist.pkl')}")

# ==================== 6. CNN vs SVM 对比分析 ====================
print("\n" + "=" * 70)
print("第6步：CNN vs SVM 综合对比")
print("=" * 70)

# --- 对比柱状图 ---
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['准确率', '精确率(加权)', '召回率(加权)', 'F1分数(加权)']

# CNN
cnn_precision = precision_score(all_labels, all_preds, average='weighted')
cnn_recall = recall_score(all_labels, all_preds, average='weighted')
cnn_f1 = f1_score(all_labels, all_preds, average='weighted')
cnn_scores = [cnn_acc, cnn_precision, cnn_recall, cnn_f1]

# SVM
svm_precision = precision_score(y_test_all, svm_test_pred, average='weighted')
svm_recall = recall_score(y_test_all, svm_test_pred, average='weighted')
svm_f1 = f1_score(y_test_all, svm_test_pred, average='weighted')
svm_scores = [svm_test_acc, svm_precision, svm_recall, svm_f1]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, cnn_scores, width, label='CNN', color='#45B7D1', edgecolor='white')
bars2 = ax.bar(x + width/2, svm_scores, width, label='SVM', color='#FF6B6B', edgecolor='white')

for bar, val in zip(bars1, cnn_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, svm_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('分值', fontsize=12)
ax.set_title('CNN vs SVM 测试集性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cnn_vs_svm_comparison.png'))
plt.close()
print("图7已保存: cnn_vs_svm_comparison.png")

# --- ROC曲线 (每类的微平均) ---
fig, ax = plt.subplots(figsize=(10, 8))
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test_all, classes=range(10))

# CNN ROC
fpr_cnn, tpr_cnn, _ = roc_curve(y_test_bin.ravel(), all_probs.ravel())
auc_cnn = auc(fpr_cnn, tpr_cnn)
ax.plot(fpr_cnn, tpr_cnn, lw=2, color='#45B7D1', label=f'CNN (micro-avg AUC={auc_cnn:.4f})')

# SVM ROC
fpr_svm, tpr_svm, _ = roc_curve(y_test_bin.ravel(), svm_test_prob.ravel())
auc_svm = auc(fpr_svm, tpr_svm)
ax.plot(fpr_svm, tpr_svm, lw=2, color='#FF6B6B', label=f'SVM (micro-avg AUC={auc_svm:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='随机猜测')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
ax.set_title('CNN vs SVM ROC曲线对比 (One-vs-Rest Micro-Average)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cnn_vs_svm_roc.png'))
plt.close()
print("图8已保存: cnn_vs_svm_roc.png")

# --- 每类准确率对比 ---
fig, ax = plt.subplots(figsize=(12, 5))
class_acc_cnn = []
class_acc_svm = []
for c in range(10):
    mask = y_test_all == c
    class_acc_cnn.append(accuracy_score(all_labels[mask], all_preds[mask]))
    class_acc_svm.append(accuracy_score(y_test_all[mask], svm_test_pred[mask]))

x = np.arange(10)
width = 0.35
ax.bar(x - width/2, class_acc_cnn, width, label='CNN', color='#45B7D1', edgecolor='white')
ax.bar(x + width/2, class_acc_svm, width, label='SVM', color='#FF6B6B', edgecolor='white')
for i, (acc1, acc2) in enumerate(zip(class_acc_cnn, class_acc_svm)):
    ax.text(i - width/2, acc1 + 0.005, f'{acc1:.3f}', ha='center', fontsize=8)
    ax.text(i + width/2, acc2 + 0.005, f'{acc2:.3f}', ha='center', fontsize=8)
ax.set_xlabel('数字类别', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('CNN vs SVM 每个数字类别的分类准确率', fontsize=14, fontweight='bold')
ax.set_xticks(range(10))
ax.legend(fontsize=11)
ax.set_ylim(0.8, 1.01)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'per_class_accuracy.png'))
plt.close()
print("图9已保存: per_class_accuracy.png")

# 保存结果摘要
summary_df = pd.DataFrame({
    'CNN': [cnn_acc, cnn_precision, cnn_recall, cnn_f1, best_val_acc/100],
    'SVM': [svm_test_acc, svm_precision, svm_recall, svm_f1, svm_train_acc]
}, index=['测试准确率', '精确率(加权)', '召回率(加权)', 'F1分数(加权)', '训练/验证准确率'])
summary_df.to_csv(os.path.join(DATA_DIR, 'results_summary.csv'), encoding='utf-8-sig')
print(f"\n结果摘要已保存: {os.path.join(DATA_DIR, 'results_summary.csv')}")

# ==================== 7. 错误案例可视化 ====================
print("\n" + "=" * 70)
print("第7步：错误案例分析")
print("=" * 70)

# CNN错误案例
wrong_mask = all_preds != all_labels
wrong_indices = np.where(wrong_mask)[0]

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i, ax in enumerate(axes.flat[:16]):
    if i < len(wrong_indices):
        idx = wrong_indices[i]
        ax.imshow(raw_test.data[idx], cmap='gray')
        ax.set_title(f'真:{all_labels[idx]} 预:{all_preds[idx]}', fontsize=9,
                     color='red', fontweight='bold')
    ax.axis('off')
plt.suptitle('CNN 错误分类案例 (真实标签 vs 预测标签)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cnn_error_cases.png'))
plt.close()
print("图10已保存: cnn_error_cases.png")

# ==================== 8. 最终结论 ====================
print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                CNN vs SVM 手写数字识别性能对比                    ║
╠══════════════╦══════════╦══════════╦══════════╦══════════════════╣
║     算法     ║  准确率  ║  精确率  ║ F1分数   ║   模型类型       ║
╠══════════════╬══════════╬══════════╬══════════╬══════════════════╣
║     CNN      ║ {cnn_acc:.4f}  ║ {cnn_precision:.4f}  ║ {cnn_f1:.4f}  ║   深度学习       ║
║     SVM      ║ {svm_test_acc:.4f}  ║ {svm_precision:.4f}  ║ {svm_f1:.4f}  ║   传统ML+PCA     ║
╚══════════════╩══════════╩══════════╩══════════╩══════════════════╝
""")

print("\n[SUCCESS] MNIST ML Pipeline 运行完成！")
print(f"   图表目录: {FIGURES_DIR}")
print(f"   模型目录: {MODELS_DIR}")
print(f"   数据目录: {DATA_DIR}")
