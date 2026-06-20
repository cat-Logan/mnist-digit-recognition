# -*- coding: utf-8 -*-
"""
《机器学习》课程设计 —— 手写数字识别系统 Web应用
运行方式: streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import joblib
import os
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# ====== 页面配置 ======
st.set_page_config(
    page_title="手写数字识别系统",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== 路径 ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====== CNN 模型 ======
class CNN(torch.nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.25):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout1 = torch.nn.Dropout2d(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# ====== 加载模型 ======
@st.cache_resource
def load_models():
    models = {}
    errors = []
    try:
        cnn_path = os.path.join(MODELS_DIR, 'cnn_mnist.pth')
        if os.path.exists(cnn_path):
            ckpt = torch.load(cnn_path, map_location=DEVICE, weights_only=False)
            cnn = CNN(**ckpt['model_config']).to(DEVICE)
            cnn.load_state_dict(ckpt['model_state_dict'])
            cnn.eval()
            models['CNN'] = cnn
        else:
            errors.append('CNN模型未找到，请先运行 ml_pipeline.py')

        for name, fname in [('svm', 'svm_mnist.pkl'), ('pca', 'pca_mnist.pkl'),
                            ('scaler', 'scaler_svm.pkl')]:
            path = os.path.join(MODELS_DIR, fname)
            if os.path.exists(path):
                models[name] = joblib.load(path)
            else:
                errors.append(f'{fname} 未找到')
        return models, errors if errors else None
    except Exception as e:
        return {}, [str(e)]


# ====== 预处理 ======
def preprocess_cnn(img_array):
    """img_array: (28,28) float [0,1], 黑底白字"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return tfm(arr).unsqueeze(0).to(DEVICE)


def preprocess_svm(img_array):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, -1)


# ====== 侧边栏 ======
st.sidebar.title("🔢 手写数字识别系统")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 选择模型")
model_choice = st.sidebar.selectbox(
    "请选择识别算法：",
    ["CNN (卷积神经网络)", "SVM (支持向量机)"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.info("""
**使用说明:**
1. 在黑色画布上用鼠标绘制数字 (0-9)
2. 数字尽量写大一点
3. 点击下方「识别」按钮查看结果
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 模型性能")
st.sidebar.markdown("""
| 算法 | 测试准确率 |
|------|:--------:|
| CNN | **99.44%** |
| SVM | 96.46% |
""")

# ====== 主页面 ======
st.title("🔢 基于CNN/SVM的手写数字识别系统")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["✍️ 手写识别", "📊 模型对比", "🔍 数据探索", "📝 关于系统"]
)

# ==================== Tab 1: 手写识别 ====================
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("✍️ 在画布上绘制数字")
        st.caption("用鼠标在黑色区域画一个数字 (0-9)")

        # ---- 可绘制画布 ----
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="mnist_canvas",
            display_toolbar=True,
            update_streamlit=True,
        )

        st.caption("画好后点击上方工具栏的清空按钮可重画")

    with col2:
        st.subheader("🔍 识别结果")

        if canvas_result.image_data is not None:
            # 从 canvas 提取图片数据
            img_data = canvas_result.image_data  # RGBA uint8 (280,280,4)
            gray = img_data[:, :, 0].astype(np.float32) / 255.0  # 取R通道 = 灰度

            # 检查是否画了内容
            has_drawing = gray.mean() > 0.02

            if has_drawing:
                # 预处理成28x28
                img_28 = np.array(Image.fromarray(
                    (gray * 255).astype(np.uint8)
                ).resize((28, 28), Image.LANCZOS)).astype(np.float32) / 255.0

                # MNIST 是黑底白字
                # canvas 是黑底白字，无需反色
                # 但如果用户画得太少，增强一下
                if img_28.max() < 0.5:
                    img_28 = img_28 / (img_28.max() + 1e-10)

                # 显示预处理后的小图
                st.image(img_28, caption='预处理后 (28x28)', width=140, clamp=True)

                # 加载模型并识别
                models, errors = load_models()

                if errors:
                    st.error(f"⚠️ {'; '.join(errors)}")
                else:
                    # 推理
                    if 'CNN' in model_choice:
                        tensor = preprocess_cnn(img_28)
                        with torch.no_grad():
                            output = models['CNN'](tensor)
                            probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    else:
                        flat = preprocess_svm(img_28)
                        flat_pca = models['pca'].transform(flat)
                        flat_scaled = models['scaler'].transform(flat_pca)
                        probs = models['svm'].predict_proba(flat_scaled)[0]

                    pred = int(np.argmax(probs))

                    # 显示结果
                    st.success(f"## 识别结果: **{pred}**")
                    st.caption(f"置信度: {probs[pred]:.2%}")

                    # 概率图
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bar_colors = ['#FF4444' if i == pred else '#4ECDC4' for i in range(10)]
                    ax.bar(range(10), probs, color=bar_colors, edgecolor='white')
                    ax.set_xlabel('数字', fontsize=12)
                    ax.set_ylabel('概率', fontsize=12)
                    ax.set_title(f'{model_choice.split()[0]} 预测概率分布', fontsize=13, fontweight='bold')
                    ax.set_xticks(range(10))
                    ax.set_ylim(0, 1)
                    for i, p in enumerate(probs):
                        if p > 0.02:
                            ax.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=8)
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("👈 请在左侧画布上绘制一个数字")
        else:
            st.info("👈 请在左侧画布上绘制一个数字")

    # 图片上传作为备选
    st.markdown("---")
    st.subheader("📤 或者上传手写数字图片")
    uploaded_file = st.file_uploader(
        "选择一张手写数字图片",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        key="upload_mnist"
    )
    if uploaded_file is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            img = Image.open(uploaded_file).convert('L')
            st.image(img, caption='原始图片', width=200)
        with col_b:
            img_28 = img.resize((28, 28), Image.LANCZOS)
            arr = np.array(img_28).astype(np.float32) / 255.0
            if arr.mean() > 0.5:
                arr = 1.0 - arr
            st.image(arr, caption='预处理后 (28x28)', width=140, clamp=True)

            models, errors = load_models()
            if errors:
                st.error(f"⚠️ {'; '.join(errors)}")
            else:
                if 'CNN' in model_choice:
                    tensor = preprocess_cnn(arr)
                    with torch.no_grad():
                        probs = F.softmax(models['CNN'](tensor), dim=1).cpu().numpy()[0]
                else:
                    flat = preprocess_svm(arr)
                    flat_pca = models['pca'].transform(flat)
                    flat_scaled = models['scaler'].transform(flat_pca)
                    probs = models['svm'].predict_proba(flat_scaled)[0]
                pred = int(np.argmax(probs))
                st.success(f"## 识别结果: **{pred}**")


# ==================== Tab 2: 模型对比 ====================
with tab2:
    st.subheader("📊 CNN vs SVM 模型性能对比")

    summary_path = os.path.join(BASE_DIR, 'data', 'results_summary.csv')
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path, index_col=0)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.warning("结果摘要未找到，请先运行 ML Pipeline")

    chart_files = [
        ('cnn_training_curves.png', 'CNN 训练曲线'),
        ('cnn_confusion_matrix.png', 'CNN 混淆矩阵'),
        ('svm_confusion_matrix.png', 'SVM 混淆矩阵'),
        ('cnn_vs_svm_comparison.png', '综合指标对比'),
        ('cnn_vs_svm_roc.png', 'ROC曲线对比'),
        ('per_class_accuracy.png', '每类准确率对比'),
    ]
    for fname, title in chart_files:
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"### {title}")
            st.image(path, use_container_width=True)


# ==================== Tab 3: 数据探索 ====================
with tab3:
    st.subheader("🔍 MNIST 数据集探索")
    st.markdown("""
    **MNIST** 包含 70,000 张 28×28 像素灰度手写数字图像。

    | 属性 | 值 |
    |------|-----|
    | 来源 | http://yann.lecun.com/exdb/mnist/ |
    | 训练集 | 60,000 张 |
    | 测试集 | 10,000 张 |
    | 图像尺寸 | 28 × 28 |
    | 类别数 | 10 (0-9) |
    """)
    for fname, title in [
        ('mnist_samples.png', '手写数字样本'),
        ('class_distribution.png', '类别分布'),
        ('mean_digits.png', '像素均值图'),
    ]:
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"### {title}")
            st.image(path, use_container_width=True)


# ==================== Tab 4: 关于系统 ====================
with tab4:
    st.subheader("📝 关于手写数字识别系统")
    st.markdown("""
    ### 🎯 项目背景
    手写数字识别是计算机视觉的基础问题，广泛应用于邮政编码自动分拣、银行票据处理等场景。

    ### 🧠 算法

    **CNN (卷积神经网络)** — 2层卷积+2层全连接，824,650参数，测试准确率 **99.44%**

    **SVM (支持向量机)** — RBF核 + PCA降维(100维)，测试准确率 **96.46%**

    ### 📚 参考文献
    [1] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proc. IEEE, 1998.
    [2] Cortes C, Vapnik V. Support-vector networks[J]. Machine Learning, 1995.
    [3] Paszke A, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]. NeurIPS, 2019.
    """)

st.markdown("---")
st.caption("《机器学习》课程设计 · 手写数字识别系统 · CNN + SVM · 基于 MNIST 数据集")
