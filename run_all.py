# -*- coding: utf-8 -*-
"""
《机器学习》课程设计 —— 手写数字识别系统
一键运行脚本

运行方式:
    python run_all.py
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(name, script_path):
    print(f"\n{'='*60}")
    print(f"  >>> 执行: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n[WARNING] {name} 返回非零退出码: {result.returncode}")
    return result

if __name__ == '__main__':
    print("=" * 60)
    print("   《机器学习》课程设计 - 手写数字识别系统")
    print("   一键运行脚本")
    print("=" * 60)

    run_step("ML Pipeline (CNN训练+SVM对比+可视化)",
             os.path.join(BASE_DIR, 'src', 'ml_pipeline.py'))

    run_step("生成综合设计报告",
             os.path.join(BASE_DIR, 'src', 'generate_report.py'))

    print(f"\n{'='*60}")
    print("   [DONE] 所有任务完成！")
    print(f"   - CNN模型: {os.path.join(BASE_DIR, 'models', 'cnn_mnist.pth')}")
    print(f"   - SVM模型: {os.path.join(BASE_DIR, 'models', 'svm_mnist.pkl')}")
    print(f"   - 可视化图表: {os.path.join(BASE_DIR, 'figures')}")
    print(f"   - 综合设计报告: {os.path.join(BASE_DIR, 'report')}")
    print(f"   - Web应用: streamlit run {os.path.join(BASE_DIR, 'src', 'app.py')}")
    print(f"{'='*60}")
