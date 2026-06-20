# 🔢 基于CNN/SVM的手写数字识别系统

## 🌐 在线体验

> **直接使用，无需安装：** https://mnist-digit-recognition-5aempm6z3nruddvey57jxz.streamlit.app/

在画布上自由绘制手写数字（0-9），在线实时识别！

---

## 📋 项目简介

本项目是《机器学习》课程综合设计作品（选题#16），基于经典的 **MNIST** 手写数字数据集，实现并对比了 **CNN（卷积神经网络）** 和 **SVM（支持向量机）+ PCA** 两种机器学习算法在图像多分类问题上的性能表现。系统包含完整的数据分析、CNN模型训练、SVM对比实验、多维评估和基于Streamlit的交互式Web应用。

## 🎯 问题定义

- **问题类型**: 图像多分类 (10类，数字0-9)
- **应用背景**: 邮政编码自动识别、银行票据处理、表单数据数字化
- **输入特征**: 28×28 灰度图像
- **输出类别**: 数字 0, 1, 2, ..., 9

## 📊 数据集

- **来源**: MNIST (http://yann.lecun.com/exdb/mnist/)
- **训练集**: 60,000 张
- **测试集**: 10,000 张
- **图像尺寸**: 28×28 灰度图
- **类别数**: 10（均衡分布）

## 🧠 算法

| 算法 | 类型 | 核心架构 |
|------|------|----------|
| **CNN** | 深度学习 | 2×Conv(32+64) + 2×FC(256+10) + BN + Dropout |
| **SVM** | 传统ML | RBF核 + PCA降维(100维) |

## 🚀 快速开始

### 安装依赖
```bash
cd mnist_digit_recognition
pip install -r requirements.txt
```

### 步骤1：训练模型
```bash
python src/ml_pipeline.py
```
执行后将：
- 自动下载 MNIST 数据集
- 训练 CNN（15 epochs, ~5分钟）
- 训练 SVM + PCA（~2分钟）
- 生成 10 张评估图表
- 保存模型到 `models/` 目录

### 步骤2：启动Web应用
```bash
streamlit run src/app.py
```
浏览器访问 `http://localhost:8501`

### 步骤3：生成报告
```bash
python src/generate_report.py
```

## 📁 项目结构

```
mnist_digit_recognition/
├── src/
│   ├── ml_pipeline.py       # CNN训练 + SVM对比 + 评估可视化
│   ├── app.py               # Streamlit Web应用 (Canvas绘画+上传)
│   └── generate_report.py   # 综合设计报告生成器
├── models/
│   ├── cnn_mnist.pth        # CNN模型权重
│   ├── cnn_mnist_traced.pt  # CNN TorchScript模型
│   ├── svm_mnist.pkl        # SVM模型
│   ├── pca_mnist.pkl        # PCA降维器
│   └── scaler_svm.pkl       # 标准化器
├── figures/                 # 10张评估图表
├── data/                    # MNIST数据集 + 结果汇总
├── report/                  # 综合设计报告
├── requirements.txt
└── README.md
```

## 📈 模型性能对比

| 指标 | CNN | SVM (PCA) |
|------|:---:|:---------:|
| 测试准确率 | **99.04%** | ~96.22% |
| F1分数 | **0.9904** | ~0.9621 |
| 参数量 | ~600K | 支持向量数 |
| 训练时间 | ~5分钟 | ~2分钟 |

## 🛠️ 技术栈

- **PyTorch** — CNN模型定义与训练
- **scikit-learn** — SVM、PCA、评估指标
- **Streamlit** — 交互式Web界面
- **Matplotlib/Seaborn** — 可视化

## 📚 参考文献

[1] LeCun Y, et al. Gradient-based learning applied to document recognition. Proc. IEEE, 1998.

[2] Cortes C, Vapnik V. Support-vector networks. Machine Learning, 1995.

[3] Paszke A, et al. PyTorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.
