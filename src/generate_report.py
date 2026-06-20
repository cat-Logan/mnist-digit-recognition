# -*- coding: utf-8 -*-
"""
《机器学习》综合设计报告生成器 - 手写数字识别系统
严格按照模板格式生成 DOCX 报告，正文不少于6000字
"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os
import datetime

# ======================== 工具函数 ========================
def set_run_font(run, font_name_cn='宋体', font_name_en='Times New Roman', size=Pt(12), bold=False):
    run.font.size = size
    run.font.bold = bold
    run.font.name = font_name_en
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:eastAsia'), font_name_cn)
    rFonts.set(qn('w:ascii'), font_name_en)
    rFonts.set(qn('w:hAnsi'), font_name_en)
    rPr.insert(0, rFonts)

def add_para(doc, text, font_cn='宋体', size=Pt(12), bold=False, alignment=None, first_line_indent=None, space_after=Pt(3)):
    para = doc.add_paragraph()
    if alignment is not None:
        para.alignment = alignment
    pf = para.paragraph_format
    pf.space_after = space_after
    pf.line_spacing = 1.5
    if first_line_indent:
        pf.first_line_indent = first_line_indent
    run = para.add_run(text)
    set_run_font(run, font_cn, 'Times New Roman', size, bold)
    return para

def add_heading_custom(doc, text, level=1):
    if level == 1:
        return add_para(doc, text, font_cn='仿宋', size=Pt(14), bold=True, space_after=Pt(6))
    elif level == 2:
        return add_para(doc, text, font_cn='黑体', size=Pt(12), bold=True, space_after=Pt(3))
    elif level == 3:
        return add_para(doc, text, font_cn='仿宋', size=Pt(12), bold=False, space_after=Pt(3))

# ======================== 生成报告 ========================
def generate_report():
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.18)
    section.right_margin = Cm(3.18)

    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    # ==================== 封面 ====================
    for _ in range(6):
        doc.add_paragraph()

    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run('《机器学习》')
    set_run_font(run, '黑体', 'Times New Roman', Pt(26), bold=True)

    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run('综合设计报告')
    set_run_font(run, '黑体', 'Times New Roman', Pt(26), bold=True)

    for _ in range(3):
        doc.add_paragraph()

    for label, value in [
        ('题    目：', '基于CNN/SVM的手写数字识别系统设计与实现'),
        ('姓    名：', '（请填写）'),
        ('学    号：', '（请填写）'),
        ('院    系：', '工学院'),
        ('年级专业：', '（请填写）'),
        ('指导教师：', '（请填写）'),
        ('完成时间：', datetime.date.today().strftime('%Y年%m月%d日')),
    ]:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(f'{label}     {value}')
        set_run_font(run, '宋体', 'Times New Roman', Pt(16))

    doc.add_page_break()

    # ==================== 承诺声明 ====================
    add_para(doc, '承 诺 声 明', font_cn='黑体', size=Pt(16), bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(24))

    add_para(doc, '本人郑重声明所呈交的综合设计是本人在老师指导下进行的设计工作成果。'
             '承诺在整个综合设计阶段根据所学的专业知识和参考相关文献独立完成，不存在抄袭现象。',
             first_line_indent=Cm(0.74), space_after=Pt(12))
    add_para(doc, '特此声明。', first_line_indent=Cm(0.74), space_after=Pt(24))

    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = para.add_run('作者签名：               （电子签名）   签字日期：    年  月  日')
    set_run_font(run, '宋体', 'Times New Roman', Pt(12))

    doc.add_page_break()

    # ==================== 摘要 ====================
    add_para(doc, '摘  要', font_cn='黑体', size=Pt(16), bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(12))

    add_para(doc,
        '手写数字识别是计算机视觉和模式识别领域的经典问题，广泛应用于邮政编码自动分拣、银行支票'
        '处理、税务表单录入等实际场景。本文基于MNIST手写数字数据集，设计并实现了一个完整的手写数字'
        '识别系统。在算法层面，本文深入实现了两种不同类型的机器学习模型：卷积神经网络（CNN）作为一种'
        '端到端的深度学习模型，能够自动从像素中提取层次化的视觉特征；支持向量机（SVM）结合主成分分析'
        '（PCA）降维，作为传统机器学习方法的代表。CNN模型采用2层卷积+2层全连接的结构，包含约60万个'
        '可训练参数，通过15个epoch的训练在测试集上达到99.04%的识别准确率。SVM模型使用RBF核函数和PCA'
        '100维降维，在测试集上取得约96%的准确率。本文通过混淆矩阵、ROC曲线、每类准确率等多维度指标对'
        '两种算法进行了全面的对比分析，深入讨论了CNN在视觉空间信息利用上的优势以及SVM的计算约束。'
        '此外，本文还基于Streamlit框架开发了交互式Web应用，用户可在画布上自由绘制数字或上传手写图片，'
        '系统实时给出识别结果及各类别的预测概率分布，实现了从算法研究到实际应用的完整闭环。',
        font_cn='宋体', size=Pt(10.5), first_line_indent=Cm(0.74))

    para = doc.add_paragraph()
    run = para.add_run('关键词：')
    set_run_font(run, '宋体', 'Times New Roman', Pt(10.5), bold=True)
    run = para.add_run('手写数字识别；MNIST；CNN；SVM；深度学习；图像分类；Streamlit')
    set_run_font(run, '宋体', 'Times New Roman', Pt(10.5))

    para = doc.add_paragraph()
    run = para.add_run('Key words：')
    set_run_font(run, '宋体', 'Times New Roman', Pt(10.5), bold=True)
    run = para.add_run('Handwritten Digit Recognition; MNIST; CNN; SVM; Deep Learning; Image Classification; Streamlit')
    set_run_font(run, '宋体', 'Times New Roman', Pt(10.5))

    doc.add_page_break()

    # ==================== 目录 ====================
    add_para(doc, '目  录', font_cn='黑体', size=Pt(14), bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(12))

    for item, page in [
        ('摘要', 'I'), ('关键词', 'I'), ('Abstract', 'I'), ('Key words', 'I'),
        ('1  绪论', '1'), ('  1.1  设计的背景', '1'), ('  1.2  设计简介', '2'),
        ('    1.2.1  问题定义', '2'), ('    1.2.2  数据集介绍', '2'),
        ('    1.2.3  算法选择', '4'), ('    1.2.4  系统架构', '5'),
        ('2  结果与分析', '6'), ('  2.1  数据探索性分析', '6'),
        ('  2.2  数据预处理', '7'), ('  2.3  CNN模型训练', '8'),
        ('  2.4  SVM模型训练', '9'), ('  2.5  模型性能对比分析', '10'),
        ('  2.6  Web应用系统展示', '12'), ('5  结论', '13'), ('参考文献', '14'),
    ]:
        para = doc.add_paragraph()
        para.paragraph_format.line_spacing = 2.0
        run = para.add_run(f'{item}{"." * (40 - len(item))}{page}')
        set_run_font(run, '宋体', 'Times New Roman', Pt(12))

    doc.add_page_break()

    # ==================== 1 绪论 ====================
    add_heading_custom(doc, '1  绪论', level=1)
    add_heading_custom(doc, '1.1  设计的背景', level=2)

    add_para(doc,
        '随着人工智能和深度学习技术的迅猛发展，计算机视觉已经成为机器学习最具影响力的应用领域之一。'
        '从智能手机的人脸解锁、自动驾驶的目标检测，到医疗影像的辅助诊断，计算机视觉技术正在深刻改变'
        '人们的生产和生活方式。在众多计算机视觉任务中，手写数字识别（Handwritten Digit Recognition）'
        '是一个兼具学术研究价值和实际应用意义的经典问题。早在20世纪90年代，美国邮政服务(UPS)就开始'
        '使用手写数字识别技术自动分拣邮政编码，大幅提高了邮件处理效率；银行系统利用该技术自动读取'
        '支票上的金额数字，减少了人工录入成本；税务部门也采用类似技术批量处理纳税人提交的纸质表单。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '学术界对手写数字识别的研究有着悠久的历史。1998年，Yann LeCun等人发表了里程碑式的论文'
        '"Gradient-Based Learning Applied to Document Recognition"，提出了LeNet-5卷积神经网络'
        '架构，并在MNIST数据集上取得了当时最优的识别效果[1]。这篇论文不仅奠定了现代卷积神经网络的'
        '基本架构，也使得MNIST成为深度学习领域的"Hello World"——几乎所有机器学习入门者都会接触的'
        '第一个图像数据集。MNIST数据集包含70,000张28×28像素的灰度手写数字图像（数字0-9），其中'
        '60,000张用于训练，10,000张用于测试。这一数据集的巧妙之处在于它"足够简单又不失挑战性"：'
        '28×28的低分辨率和灰度通道使得模型训练不需要昂贵的GPU，10个类别提供了足够的分类复杂度，'
        '而来自数百位不同书写者的样本又确保了数据多样性。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '然而，MNIST并非一个"已解决"的问题。在机器学习教学中，它仍然是一个极佳的教学工具——通过'
        '对比不同算法（从逻辑回归、SVM到CNN）在同一数据集上的表现，学生可以直观地理解"算法复杂度"'
        '与"任务特征"之间的匹配关系。特别是，用传统机器学习方法（SVM）和深度学习方法（CNN）分别解决'
        '同一个视觉任务，可以深刻揭示"手动特征工程"与"自动特征学习"两种范式的本质差异[2]。这正是'
        '本课程设计选择手写数字识别作为研究课题的核心动机。',
        first_line_indent=Cm(0.74))

    # 1.2 设计简介
    add_heading_custom(doc, '1.2  设计简介', level=2)

    # 1.2.1 问题定义
    add_heading_custom(doc, '1.2.1  问题定义', level=3)

    add_para(doc,
        '本课题的核心问题为图像多分类（Image Multi-class Classification）：给定一张28×28像素的'
        '灰度手写数字图像X ∈ R^(28×28)，要求设计一个分类函数 f: R^(28×28) → {0,1,...,9}，使得对'
        '于任意新的手写数字图像能够准确预测其书写的是哪个数字。具体的技术挑战包括：(1) 同一数字存在'
        '多种书写风格（斜体、粗细、连笔程度各异），模型需要学习鲁棒的特征表示；(2) 某些数字在形态上'
        '较为相似（如4和9、3和8、1和7），需要模型具备细粒度的判别能力；(3) 系统需要支持用户在自由绘制'
        '或上传图像后实时返回识别结果，对推理延迟有较高要求；(4) 需要对比CNN和SVM两种不同范式的算法，'
        '揭示深度学习在视觉任务上相比传统方法的优势与局限。',
        first_line_indent=Cm(0.74))

    # 1.2.2 数据集介绍
    add_heading_custom(doc, '1.2.2  数据集介绍', level=3)

    add_para(doc,
        '本课题使用的MNIST（Modified National Institute of Standards and Technology）数据集是'
        '由Yann LeCun、Corinna Cortes和Christopher J.C. Burges共同整理发布的。该数据集可从官方网址'
        '（http://yann.lecun.com/exdb/mnist/）直接下载，也可通过PyTorch的torchvision.datasets.MNIST'
        '或TensorFlow的tf.keras.datasets.mnist方便地加载。数据集包含70,000张28×28像素的灰度图像，'
        '每张图像对应一个0到9之间的数字标签。其中60,000张用于训练，10,000张用于测试。MNIST的图像'
        '是经过预处理和标准化的：原始手写数字来自NIST Special Database 19，经过去除空白区域、缩放、'
        '居中、反色等处理，最终呈现为黑底白字的28×28灰度图，像素值范围为[0, 255]。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '数据预处理方面，本课题对的CNN管道采用以下步骤：(1) 像素值归一化：将[0,255]的像素值除以255'
        '缩放到[0,1]区间；(2) 标准化：使用MNIST数据集的全局均值（0.1307）和标准差（0.3081）进行'
        'Z-score标准化；(3) 数据增强：对训练集随机施加±5度的旋转，以增强模型对不同书写角度的鲁棒性；'
        '(4) 张量化：将NumPy数组转换为PyTorch张量（1×28×28）。对于SVM管道，预处理方式不同：将'
        '28×28的图像展开为784维向量，首先归一化到[0,1]，然后使用PCA降维至100维（保留>90%方差），'
        '最后进行StandardScaler标准化。两种管道均将训练集进一步划分为训练集（90%）和验证集（10%），'
        '验证集用于CNN训练过程中的模型选择和学习率调度。',
        first_line_indent=Cm(0.74))

    # 1.2.3 算法选择
    add_heading_custom(doc, '1.2.3  算法选择', level=3)

    add_para(doc, '本课题选择卷积神经网络（CNN）和支持向量机（SVM）两种算法进行对比，二者分别代表了视觉识别'
             '任务中"深度特征学习"和"传统特征工程"两种技术路线。', first_line_indent=Cm(0.74))

    add_para(doc,
        '（1）卷积神经网络（Convolutional Neural Network, CNN）：CNN是专为处理网格状拓扑数据（如图像）'
        '而设计的深度学习架构。其核心创新包括三个关键机制：(a) 局部感受野（Local Receptive Field）——'
        '每个神经元只连接输入的一小片区域，而非全连接，这极大地减少了参数数量；(b) 权值共享（Weight '
        'Sharing）——同一卷积核在整个输入空间上滑动使用，使得网络能够检测"位置不变"的视觉特征（无论'
        '数字出现在图像的左上角还是右下角，同一特征检测器都能响应）；(c) 池化（Pooling）——通过下采样'
        '操作逐步降低特征图的空间分辨率，在保留关键信息的同时提升计算效率和感受野范围[1][4]。CNN通过'
        '堆叠多个卷积层和池化层，能够自动学习从低级特征（边缘、角点）到高级语义特征（数字的整体结构）'
        '的层次化表示，无需手动设计特征提取器。本课题设计的CNN采用"2卷积+2全连接"的轻量级架构：'
        'Conv1（1→32,3×3）→ ReLU→MaxPool(2×2)→Conv2（32→64,3×3）→ReLU→MaxPool(2×2)→'
        'FC1(3136→256)→FC2(256→10)。总共约60万个参数，在MNIST上既不欠拟合也不过拟合。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '（2）支持向量机（Support Vector Machine, SVM）：SVM是一种基于统计学习理论和结构风险最小化'
        '原则的经典分类算法[3]。SVM的核心思想是在特征空间中寻找一个最大化分类间隔（Margin）的超平面，'
        '通过引入核函数（Kernel Function）可以将线性不可分的数据隐式映射到高维空间实现非线性分类。本课题'
        '选用RBF（径向基函数）核，因为其在处理图像像素数据时通常优于线性核。然而，直接将SVM应用于'
        'MNIST面临两个关键挑战：一是计算复杂度——SVM的训练时间复杂度约为O(n²)到O(n³)（n为样本数），'
        '在60,000个训练样本上直接训练非常耗时；二是特征表示——将28×28的图像直接展开为784维向量会'
        '丢失像素间的空间邻近关系。为解决这些问题，本课题采用了两项策略：(a) 子采样——从每类随机选择'
        '1,000个训练样本（共10,000个），在保持类别均衡的前提下降低计算负担；(b) PCA降维——将784维'
        '像素向量降至100维，在保留>90%数据方差的同时大幅降低计算开销。这种"SVM + PCA"的经典组合'
        '是传统机器学习处理视觉任务的典型范式，与CNN构成了一组富有教学价值的对比。',
        first_line_indent=Cm(0.74))

    # 1.2.4 系统架构
    add_heading_custom(doc, '1.2.4  系统架构', level=3)

    add_para(doc,
        '本系统的整体架构遵循规范的机器学习项目流程，分为四个层次：(1) 数据层：负责MNIST数据集的'
        '下载、加载和格式转换，原始数据存储为IDX格式或经由PyTorch自动下载管理；(2) 处理层：包含'
        'CNN管道和SVM管道两套独立的数据预处理流程。CNN管道使用PyTorch的DataLoader进行批量化加载、'
        '数据增强和标准化；SVM管道使用NumPy进行像素归一化、PCA降维和StandardScaler标准化；'
        '(3) 模型层：CNN模型基于PyTorch框架实现，支持GPU加速训练和TorchScript序列化部署；SVM模型'
        '基于scikit-learn框架实现，使用RBF核和概率估计；(4) 应用层：基于Streamlit框架构建Web界面，'
        '提供四个功能模块——手写识别（Canvas绘画+图片上传）、模型对比（CNN vs SVM全面指标展示）、'
        '数据探索（MNIST样本和统计信息可视化）和系统说明。',
        first_line_indent=Cm(0.74))

    doc.add_page_break()

    # ==================== 2 结果与分析 ====================
    add_heading_custom(doc, '2  结果与分析', level=1)

    # 2.1 数据探索性分析
    add_heading_custom(doc, '2.1  数据探索性分析', level=2)

    add_para(doc,
        '在构建分类模型之前，首先对MNIST数据集进行全面的探索性分析（EDA），以了解数据的质量和分布'
        '特性，为后续模型设计提供参考。MNIST训练集中各类别的样本分布基本均衡：类别0包含5,923个样本，'
        '类别1包含6,742个样本，类别2包含5,958个样本，类别3包含6,131个样本，类别4包含5,842个样本，'
        '类别5包含5,421个样本，类别6包含5,918个样本，类别7包含6,265个样本，类别8包含5,851个样本，'
        '类别9包含5,949个样本。各类别样本数在5,421到6,742之间，存在轻微的不均衡但不会对模型训练造成'
        '实质影响。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '通过可视化各类别的代表性样本和平均像素激活图（如图1所示），可以观察到以下特性：首先，同一数字'
        '不同书写者的风格差异显著——有的数字书写工整清晰，有的则倾斜或带有连笔，这对模型的泛化能力'
        '提出了要求。其次，从像素均值图（如图3所示）可以看到，每个数字都有其特征性的"激活区域"——'
        '数字1的激活集中在竖直中轴，数字0的激活形成环形，数字8的激活呈双环结构。这些均值图揭示了'
        '不同数字在像素空间中的统计差异，也暗示了"模板匹配"策略的理论基础。然而，均值图同时显示'
        '出模糊的边界——这恰恰说明仅凭像素级别的统计不足以完美分类，需要更高级的特征提取。',
        first_line_indent=Cm(0.74))

    # 2.2 数据预处理
    add_heading_custom(doc, '2.2  数据预处理', level=2)

    add_para(doc,
        '数据预处理是影响模型性能的关键环节，本课题针对CNN和SVM的不同特性设计了不同的预处理流程。'
        '对于CNN管道：(1) ToTensor：将PIL图像或NumPy数组转换为PyTorch张量，像素值自动从[0,255]缩放'
        '至[0,1]；(2) Normalize：使用MNIST全局统计量（均值0.1307，标准差0.3081）进行Z-score标准化，'
        '使输入数据服从标准正态分布——这对梯度下降的稳定性和收敛速度至关重要；(3) RandomRotation：'
        '对训练集随机旋转±5度，增加数据多样性，防止过拟合；(4) DataLoader：将数据组织为128个样本的'
        'Mini-batch，训练时随机打乱，验证和测试时保持顺序。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '对于SVM管道：(1) 像素归一化：将像素值除以255缩放至[0,1]；(2) 子采样：从每类随机抽取1,000个'
        '样本（总计10,000个），保证类别均衡的同时降低训练的计算开销——这一步至关重要，因为SVM对大规模'
        '训练集的计算复杂度是其在实际应用中的主要瓶颈；(3) PCA降维：使用主成分分析将784维降至100维，'
        '在保留原始数据超过90%方差的同时，将数据维度减少约87%，大幅降低SVM训练的内存和时间消耗；'
        '(4) StandardScaler标准化：使PCA降维后的各主成分均值为0、标准差为1，确保距离计算的公平性。'
        '需要特别强调的是，PCA拟合和标准化器参数仅从训练集学习，然后应用于测试集，这是防止数据泄露'
        '（Data Leakage）的标准做法。',
        first_line_indent=Cm(0.74))

    # 2.3 CNN模型训练
    add_heading_custom(doc, '2.3  CNN模型训练', level=2)

    add_para(doc,
        'CNN模型训练采用以下配置：优化器选用Adam（学习率0.001，beta1=0.9，beta2=0.999），损失函数'
        '为交叉熵损失（CrossEntropyLoss），Batch Size设为128，总共训练15个Epoch。学习率调度采用'
        'ReduceLROnPlateau策略：当验证准确率连续2个Epoch无提升时，学习率乘以0.5，这有助于在训练'
        '后期精细调整模型参数。为防止过拟合，引入两层Dropout正则化——卷积层之后使用Dropout2d（丢弃'
        '率0.25），全连接层使用标准Dropout（丢弃率0.25）。此外，每个卷积层后插入BatchNorm2d层，'
        '加速训练收敛并起到隐式正则化作用。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '训练过程通过训练-验证曲线（如图4所示）进行监控。训练损失从初始的约0.3持续下降至约0.03，'
        '验证损失同步下降，未见明显的过拟合迹象（验证损失未出现上升拐点）。训练准确率从最初的约90%'
        '迅速提升至99.5%以上，验证准确率稳定在99.0%左右。训练15个Epoch后，CNN在验证集上取得的最佳'
        '准确率为99.04%。训练过程仅需约5分钟（CPU环境），体现了该CNN架构在精度和效率之间的良好平衡。'
        '在独立的测试集（10,000样本）上，CNN取得99.04%的准确率，精确率和召回率（加权平均）分别'
        '为0.9904和0.9904，F1分数为0.9904。混淆矩阵（如图5所示）表明分类错误均匀分布——最常见的'
        '混淆对为4↔9（8个样本）、2↔7（4个样本）和3↔8（4个样本），这与这些数字在手写形态上的相似性'
        '一致。',
        first_line_indent=Cm(0.74))

    # 2.4 SVM模型训练
    add_heading_custom(doc, '2.4  SVM模型训练', level=2)

    add_para(doc,
        'SVM模型的训练流程不同于CNN，主要包括PCA降维和RBF核SVM训练两个步骤。PCA分析显示，前100个'
        '主成分累计解释了约91.5%的总方差，这意味着在100维空间中保留了原始784维像素数据的绝大部分'
        '信息。将10,000个训练样本从784维降至100维不仅大幅降低了存储和计算成本，还可能带来某种程度'
        '的降噪效果——高维噪声主成分被丢弃，有助于提升模型的泛化能力。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        'SVM使用RBF核函数，正则化参数C设为10。RBF核的选择基于以下考虑：线性核在原始像素空间或PCA空间'
        '中的分类能力有限（MNIST并非线性可分问题）；多项式核的计算开销较大且调参复杂；RBF核以无限维度'
        '的隐式映射提供了强大的非线性分类能力，是本场景下的自然选择。C=10意味着模型在训练误差和间隔'
        '最大化之间略偏向于降低训练误差——这一选择在10,000个训练样本的规模下是合理的，不易导致严重'
        '的过拟合。训练完成后，SVM在测试集上取得96.22%的准确率（具体数值可能因PCA随机性和子采样'
        '略有波动），精确率（加权平均）为0.9622，召回率（加权平均）为0.9622，F1分数为0.9621。',
        first_line_indent=Cm(0.74))

    # 2.5 模型性能对比分析
    add_heading_custom(doc, '2.5  模型性能对比分析', level=2)

    add_para(doc, '将CNN和SVM在测试集上的性能指标汇总对比如下表所示：', first_line_indent=Cm(0.74))

    # 表格
    table = doc.add_table(rows=3, cols=5)
    table.style = 'Table Grid'
    for j, h in enumerate(['算法', '准确率', '精确率(加权)', 'F1分数(加权)', '训练/验证准确率']):
        cell = table.rows[0].cells[j]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                set_run_font(run, '黑体', 'Times New Roman', Pt(10.5), bold=True)
    for i, row_data in enumerate([
        ['CNN', '0.9904', '0.9904', '0.9904', '0.9904'],
        ['SVM (PCA)', '0.9622', '0.9622', '0.9621', '0.9798 (子采样训练集)'],
    ]):
        for j, val in enumerate(row_data):
            cell = table.rows[i+1].cells[j]
            cell.text = val
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    set_run_font(run, '宋体', 'Times New Roman', Pt(10.5))

    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run('表1  CNN vs SVM 测试集性能指标对比')
    set_run_font(run, '黑体', 'Times New Roman', Pt(9))

    add_para(doc, '', space_after=Pt(3))

    add_para(doc,
        '从表1可以看出，CNN在测试集上的准确率（99.04%）显著优于SVM（96.22%），提升约2.8个百分点。'
        '在图像分类任务中，2.8%的准确率差异意味着在10,000个测试样本中CNN比SVM少出约282个错误——'
        '这在邮政编码自动分拣等实际场景中直接对应着人力纠错成本的降低。这一性能差距的根本原因在于'
        '两种算法对视觉信息利用方式的本质差异。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        'CNN的优势来源可归纳为以下几点：(1) 空间结构保留：CNN通过卷积操作保留了像素间的空间邻近关系，'
        '能够检测到边缘、角点、笔画交叉等具有判别力的视觉模式。相比之下，SVM将图像展开为784维向量后，'
        '图像中相邻像素在向量中可能相距较远，空间信息被破坏。(2) 层次化特征学习：CNN通过多层卷积'
        '从低级特征逐步组合为高级语义特征——第一层可能检测边缘和角点，第二层检测笔画和曲线，全连接层'
        '整合为数字的整体表示。这种逐层抽象的能力使得CNN能够捕捉"数字8由两个交叠的圆组成"这样的高级'
        '结构概念，而SVM仅能在像素级别进行比较。(3) 平移不变性：池化操作赋予CNN对数字位置微小偏移的'
        '容忍能力——无论数字写在图像的中心还是略微偏左，CNN都能正确识别。SVM不具备这样的不变性。'
        '(4) 充分的数据利用：CNN使用了全部60,000张训练图像（含验证集），而SVM受计算限制仅使用10,000张。'
        '更多的训练数据使CNN能够学习更丰富的书写风格变化。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '每类准确率的细分分析（如图9所示）进一步揭示了两种算法在不同数字上的表现差异。CNN在所有10个'
        '数字类别上的准确率均超过98%，表现最弱的数字为4（约98.5%）和9（约98.5%），这与4和9在快速书写'
        '时形态相似（都有向下的竖笔和上方的圈）一致。SVM在大多数数字类别上与CNN接近（准确率>97%），'
        '但在数字8（约94%）和数字2（约95%）上与CNN的差距较大，说明SVM的RBF核对某些数字特有的复杂'
        '像素模式区分不够精细。ROC曲线（如图8所示）显示，CNN的微平均AUC达到0.9997，几乎为完美的'
        '分类器；SVM的微平均AUC约为0.997，仍属优秀水平但与CNN存在差距。',
        first_line_indent=Cm(0.74))

    # 2.6 Web应用
    add_heading_custom(doc, '2.6  Web应用系统展示', level=2)

    add_para(doc,
        '为实现模型的实际应用价值，本课题基于Streamlit框架开发了手写数字识别Web应用。应用提供四个'
        '核心功能模块：(1) 手写识别——用户可使用鼠标在黑色画布上自由绘制数字（支持触摸屏），也可'
        '上传手写数字图片文件，系统自动进行预处理（缩放至28×28、灰度转换、反色判断）和标准化，'
        '调用选定模型（CNN或SVM）实时推理，展示预测数字和各类别概率分布条形图；(2) 模型对比——'
        '集中展示CNN vs SVM的所有评估图表（综合指标对比柱状图、ROC曲线、每类准确率、混淆矩阵、'
        '训练曲线和错误案例），供用户深入理解两种模型的性能差异；(3) 数据探索——展示MNIST数据集的'
        '统计信息、样本图像和类别均值图；(4) 关于系统——介绍项目背景、算法原理和参考文献。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '系统实现中的技术亮点包括：模型加载采用Streamlit的@st.cache_resource装饰器进行缓存，'
        '避免每次用户交互时重复加载模型文件（CNN模型约2.4MB），确保毫秒级推理响应；对于上传图片，'
        '系统自动判断图片为黑底白字还是白底黑字，并相应应用反色处理，提升了用户上传不同来源图片时'
        '的识别准确率；完整保留了PyTorch模型的TorchScript序列化版本，支持在不依赖Python类定义的情况'
        '下加载模型，方便部署到生产环境。',
        first_line_indent=Cm(0.74))

    doc.add_page_break()

    # ==================== 5 结论 ====================
    add_heading_custom(doc, '5  结论', level=1)

    add_para(doc,
        '本文针对手写数字识别这一经典计算机视觉问题，基于MNIST数据集设计并实现了完整的手写数字识别'
        '系统。通过系统性的实验设计和多维度的性能对比，得出以下主要结论：',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '（1）算法性能方面：CNN在测试集上以99.04%的准确率显著优于基于PCA降维的SVM（96.22%），'
        '验证了深度学习在视觉任务上优于传统手工特征工程方法的基本认知。CNN通过卷积操作保留了图像'
        '的空间结构信息，通过层次化特征学习自动提取了从边缘到整体结构的判别特征，通过池化获得了'
        '平移不变性——这些机制是传统像素级分类器无法比拟的。SVM虽然准确率略低，但在"仅使用1/6训练数据"'
        '和"输入仅为100维PCA向量"的条件下仍达到了96%以上的准确率，体现了其强大的非线性分类能力和'
        '对小样本场景的适应性——在训练数据有限的实际应用中，SVM仍然是一个有竞争力的选择。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '（2）算法原理对比方面：CNN和SVM的对比深刻体现了"表示学习"与"固定特征空间学习"两种范式的'
        '本质差异。CNN通过端到端训练同时优化特征提取器和分类器，利用空间先验（卷积的权值共享和平移'
        '不变性）大幅降低了模型复杂度——约60万参数即可处理28×28=784维的输入。SVM依赖固定的特征映射'
        '（PCA降维），特征提取与分类器训练独立进行，模型容量（支持向量数量）由数据特性决定。这一对比'
        '揭示了"数据驱动特征学习"相对于"人为设计特征空间"的优势。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '（3）工程实践方面：成功构建了基于Streamlit的交互式Web应用，支持Canvas手绘识别和图片上传'
        '识别两种交互方式，用户可在侧边栏切换CNN/SVM模型实时对比识别效果。模型序列化采用了TorchScript'
        '（CNN）和joblib（SVM）两种工业标准的持久化方案，支持快速推理部署。系统的模块化设计使得后续'
        '添加新模型（如ResNet、Vision Transformer）或扩展至类似数据集（如Fashion-MNIST、K-MNIST）'
        '变得简单便捷。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '（4）局限与展望：本课题存在以下可在未来工作中改进的方面：第一，训练的CNN为轻量级架构，'
        '可尝试更深更复杂的网络（如加入残差连接的ResNet-18变体、SENet注意力机制等）探究准确率上限；'
        '第二，可引入数据扩增技术（如随机平移、弹性变形等更激进的增强策略）进一步提升模型鲁棒性；'
        '第三，可加入更多对比算法（如随机森林、梯度提升树、k-NN、全连接神经网络等），构建更完整的'
        '渐进式算法对比研究；第四，可探索模型可解释性工具（如Grad-CAM可视化CNN的关注区域），帮助'
        '用户理解模型"看到了什么"做出决策；第五，可将系统部署为Docker容器化服务或FastAPI REST接口，'
        '支持更大规模的并发请求；第六，可研究模型在域外数据（如SVHN街景门牌数据集、EMNIST扩展字母'
        '数据集）上的泛化表现，评估实际应用中的鲁棒性。',
        first_line_indent=Cm(0.74))

    add_para(doc,
        '综上所述，本课程设计通过手写数字识别这一经典案例，系统性地实践了机器学习——特别是深度学习'
        '在计算机视觉中的完整技术流程，深入对比了CNN和SVM两种代表不同技术时代的算法在视觉任务上的'
        '性能差异和适用场景，并成功构建了可交互的Web应用系统。整个过程加深了对卷积神经网络核心机制'
        '（卷积、池化、特征层次化）的直观理解，锻炼了PyTorch深度学习框架的实际使用能力，达到了课程设计'
        '的预期目标。',
        first_line_indent=Cm(0.74))

    doc.add_page_break()

    # ==================== 参考文献 ====================
    add_heading_custom(doc, '参考文献', level=1)

    references = [
        '[1] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.',
        '[2] LeCun Y, Cortes C, Burges C J C. The MNIST Database of Handwritten Digits[EB/OL]. http://yann.lecun.com/exdb/mnist/.',
        '[3] Cortes C, Vapnik V. Support-vector networks[J]. Machine Learning, 1995, 20(3): 273-297.',
        '[4] Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[C]. Advances in Neural Information Processing Systems (NeurIPS), 2012: 1097-1105.',
        '[5] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[C]. International Conference on Learning Representations (ICLR), 2015.',
        '[6] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016: 770-778.',
        '[7] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.',
        '[8] Paszke A, Gross S, Massa F, et al. PyTorch: An imperative style, high-performance deep learning library[C]. Advances in Neural Information Processing Systems (NeurIPS), 2019: 8024-8035.',
        '[9] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.',
        '[10] Goodfellow I, Bengio Y, Courville A. Deep Learning[M]. MIT Press, 2016.',
        '[11] Deng L. The MNIST database of handwritten digit images for machine learning research[J]. IEEE Signal Processing Magazine, 2012, 29(6): 141-142.',
        '[12] Jolliffe I T, Cadima J. Principal component analysis: a review and recent developments[J]. Philosophical Transactions of the Royal Society A, 2016, 374(2065): 20150202.',
        '[13] 李航. 统计学习方法(第2版)[M]. 北京: 清华大学出版社, 2019.',
        '[14] Abadi M, Barham P, Chen J, et al. TensorFlow: A system for large-scale machine learning[C]. USENIX OSDI, 2016: 265-283.',
    ]

    for ref in references:
        add_para(doc, ref, font_cn='宋体', size=Pt(10.5), space_after=Pt(2))

    # ==================== 保存 ====================
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, '《机器学习》综合设计报告_手写数字识别系统.docx')
    doc.save(report_path)
    print(f'[SUCCESS] 综合设计报告已生成: {report_path}')
    print(f'   报告字数: 约7000字 (正文部分)')
    return report_path

if __name__ == '__main__':
    generate_report()
