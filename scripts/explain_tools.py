import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
# --- 解決方案 1: 重設 Matplotlib 設定以消除字體警告 ---
# 這行程式碼會將 Matplotlib 的設定恢復到預設狀態，避免因環境配置問題尋找不存在的字體。
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import io
from PIL import Image
import numpy as np

# 获取整合的shap

def get_combine_shap(drug_shap_2,mid_shap, feature_prefix, feature_num=768):
    # 提取 mid_values
    mid_values = mid_shap[mid_shap['feature_name'].str.startswith(feature_prefix)]['shap'].values[:feature_num]
    
    # 提取第一层 SHAP 矩阵
    ds_matrix = drug_shap_2.iloc[:, :feature_num].values
    
    # 【关键修改】：不再计算 signed_ratio，不再除以 abs_sum
    # 直接将第一层的贡献（强度）乘以第二层的贡献（权重）
    # 这种做法下，各子模型之间的原始 SHAP 强度得以保留，具有可比性
    combined_shap = np.dot(ds_matrix, mid_values)
    
    return combined_shap


def mol_picture(shap_values,tokenized_smiles,save_path=None,AtomIndices=True):
    # 1. Reconstruct the SMILES string
    full_smiles = "".join(tokenized_smiles)
    print(f"Reconstructed SMILES string: {full_smiles}")

    # 2. Generate RDKit molecule object
    mol = Chem.MolFromSmiles(full_smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string, could not generate molecule.")

    num_atoms = mol.GetNumAtoms()

    # 3. Map SHAP values to atoms
    atom_to_shap_map = {}
    atom_idx_counter = 0
    for token, shap_val in zip(tokenized_smiles, shap_values):
        for char in token:
            if char.isalpha(): 
                if atom_idx_counter < num_atoms:
                    atom_to_shap_map[atom_idx_counter] = shap_val
                    atom_idx_counter += 1

    print(f"Successfully mapped SHAP values to {len(atom_to_shap_map)} out of {num_atoms} atoms.")

    # --- 解決方案 2: 美化繪圖並合併為單一圖片 ---

    # 4. Color mapping
    if not atom_to_shap_map:
        print("No mappable SHAP values found. Cannot generate image.")
    else:
        # --- 步骤 1: 导入 mcolors ---
        # 这一步您已经在代码中完成了： from matplotlib import colors as mcolors

        # --- 步骤 2: 定义您的自定义颜色 ---
        CUSTOM_NEG_COLOR = '#008bfb'  # 深绿色（代表负贡献）
        CUSTOM_ZERO_COLOR = '#FFFFFF' # 浅灰色或白色（代表接近零的贡献）
        CUSTOM_POS_COLOR = '#FF0000'  # 深红色（代表正贡献）

        # --- 步骤 3: 替换原有的颜色映射代码 ---
        # 找到原始代码中的这一段：
        # cmap = plt.get_cmap('bwr')

        # 替换为以下代码：
        # 定义颜色列表：[最小值颜色, 中值颜色, 最大值颜色]
        custom_colors_list = [CUSTOM_NEG_COLOR, CUSTOM_ZERO_COLOR, CUSTOM_POS_COLOR]

        # 创建自定义的颜色映射对象
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'CustomShapMap', # 颜色图的名称
            custom_colors_list,
            N=256 # 颜色级别
        )

        # 使用自定义的颜色映射对象替换原有的 'bwr'
        #cmap = custom_cmap
        cmap = plt.get_cmap('bwr')
        max_abs_val = np.max(np.abs(list(atom_to_shap_map.values())))
        norm = mcolors.Normalize(vmin=-max_abs_val, vmax=max_abs_val)
        atom_colors = {idx: cmap(norm(val)) for idx, val in atom_to_shap_map.items()}

        # 5. Generate the molecule drawing in memory
        d = rdMolDraw2D.MolDraw2DCairo(800, 600) # 增加繪圖畫布大小以提高內部品質
        draw_options = d.drawOptions()
        # --- 美化選項 ---
        draw_options.addAtomIndices = AtomIndices      # 不顯示原子索引號
        draw_options.bondLineWidth = 2           # 設定鍵結寬度
        draw_options.fillHighlights = True      # 正確：啟用實心填滿
        draw_options.highlightRadius = 0.35     # 正確：直接賦值來設定半徑
        draw_options.clearBackground = False     # 設定背景為透明

        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, 
                                        highlightAtoms=list(atom_to_shap_map.keys()),
                                        highlightAtomColors=atom_colors)
        d.FinishDrawing()
        png_data = d.GetDrawingText()
        molecule_image = Image.open(io.BytesIO(png_data))

        # 6. Create a combined plot using Matplotlib
        fig = plt.figure(figsize=(8, 8))
        # 使用 GridSpec 精確控制佈局，頂部90%空間給分子，底部10%給圖例
        gs = GridSpec(2, 1, height_ratios=[9, 1], hspace=0.05)
        
        # 上方 subplot 用於放置分子圖
        ax_mol = fig.add_subplot(gs[0])
        ax_mol.imshow(molecule_image)
        ax_mol.axis('off') # 隱藏座標軸

        # 下方 subplot 用於放置顏色圖例
        ax_legend = fig.add_subplot(gs[1])
        cb = matplotlib.colorbar.ColorbarBase(ax_legend, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_label('SHAP Value (Contribution to Prediction)', fontsize=12)
        
        fig.suptitle('SHAP Contribution on Molecular Structure', fontsize=16, y=0.95)
        plt.show()
        # 7. Save the final combined image
        #output_filename = save_path
        # 儲存為高解析度圖片 (300 DPI)
        if save_path is None:
            print("No save path provided. Skipping image saving.")
        else:
            fig.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

        #print("\n--- Execution Complete ---")
        #print(f"A single, beautified image has been saved as: {save_path}")

from rdkit.Chem import rdDepictor
def mol_picture2(shap_values,tokenized_smiles,save_path=None,AtomIndices=True):
    # 1. Reconstruct the SMILES string
    full_smiles = "".join(tokenized_smiles)
    print(f"Reconstructed SMILES string: {full_smiles}")

    # 2. Generate RDKit molecule object
    
    mol = Chem.MolFromSmiles(full_smiles)

    # 1. 生成 2D 坐标（如果 SMILES 还没有坐标）
    rdDepictor.Compute2DCoords(mol)

    # 2. 获取分子的构象 (Conformer) 并翻转 Y 轴
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        # 将 y 坐标取负，实现上下翻转
        conf.SetAtomPosition(i, (pos.x, -pos.y, pos.z))

    if mol is None:
        raise ValueError("Invalid SMILES string, could not generate molecule.")

    num_atoms = mol.GetNumAtoms()

    # 3. Map SHAP values to atoms
    atom_to_shap_map = {}
    atom_idx_counter = 0
    for token, shap_val in zip(tokenized_smiles, shap_values):
        for char in token:
            if char.isalpha(): 
                if atom_idx_counter < num_atoms:
                    atom_to_shap_map[atom_idx_counter] = shap_val
                    atom_idx_counter += 1

    print(f"Successfully mapped SHAP values to {len(atom_to_shap_map)} out of {num_atoms} atoms.")

    # --- 解決方案 2: 美化繪圖並合併為單一圖片 ---

    # 4. Color mapping
    if not atom_to_shap_map:
        print("No mappable SHAP values found. Cannot generate image.")
    else:
        # --- 步骤 1: 导入 mcolors ---
        # 这一步您已经在代码中完成了： from matplotlib import colors as mcolors

        # --- 步骤 2: 定义您的自定义颜色 ---
        CUSTOM_NEG_COLOR = '#008bfb'  # 深绿色（代表负贡献）
        CUSTOM_ZERO_COLOR = '#FFFFFF' # 浅灰色或白色（代表接近零的贡献）
        CUSTOM_POS_COLOR = '#FF0000'  # 深红色（代表正贡献）

        # --- 步骤 3: 替换原有的颜色映射代码 ---
        # 找到原始代码中的这一段：
        # cmap = plt.get_cmap('bwr')

        # 替换为以下代码：
        # 定义颜色列表：[最小值颜色, 中值颜色, 最大值颜色]
        custom_colors_list = [CUSTOM_NEG_COLOR, CUSTOM_ZERO_COLOR, CUSTOM_POS_COLOR]

        # 创建自定义的颜色映射对象
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'CustomShapMap', # 颜色图的名称
            custom_colors_list,
            N=256 # 颜色级别
        )

        # 使用自定义的颜色映射对象替换原有的 'bwr'
        #cmap = custom_cmap
        cmap = plt.get_cmap('bwr')
        max_abs_val = np.max(np.abs(list(atom_to_shap_map.values())))
        norm = mcolors.Normalize(vmin=-max_abs_val, vmax=max_abs_val)
        atom_colors = {idx: cmap(norm(val)) for idx, val in atom_to_shap_map.items()}

        # 5. Generate the molecule drawing in memory
        d = rdMolDraw2D.MolDraw2DCairo(800, 600) # 增加繪圖畫布大小以提高內部品質
        draw_options = d.drawOptions()
        # --- 美化選項 ---
        draw_options.addAtomIndices = AtomIndices      # 不顯示原子索引號
        draw_options.bondLineWidth = 2           # 設定鍵結寬度
        draw_options.fillHighlights = True      # 正確：啟用實心填滿
        draw_options.highlightRadius = 0.35     # 正確：直接賦值來設定半徑
        draw_options.clearBackground = False     # 設定背景為透明

        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, 
                                        highlightAtoms=list(atom_to_shap_map.keys()),
                                        highlightAtomColors=atom_colors)
        d.FinishDrawing()
        png_data = d.GetDrawingText()
        molecule_image = Image.open(io.BytesIO(png_data))

        # 6. Create a combined plot using Matplotlib
        fig = plt.figure(figsize=(8, 8))
        # 使用 GridSpec 精確控制佈局，頂部90%空間給分子，底部10%給圖例
        gs = GridSpec(2, 1, height_ratios=[9, 1], hspace=0.05)
        
        # 上方 subplot 用於放置分子圖
        ax_mol = fig.add_subplot(gs[0])
        ax_mol.imshow(molecule_image)
        ax_mol.axis('off') # 隱藏座標軸

        # 下方 subplot 用於放置顏色圖例
        ax_legend = fig.add_subplot(gs[1])
        cb = matplotlib.colorbar.ColorbarBase(ax_legend, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_label('SHAP Value (Contribution to Prediction)', fontsize=12)
        
        fig.suptitle('SHAP Contribution on Molecular Structure', fontsize=16, y=0.95)
        plt.show()
        # 7. Save the final combined image
        #output_filename = save_path
        # 儲存為高解析度圖片 (300 DPI)
        if save_path is None:
            print("No save path provided. Skipping image saving.")
        else:
            fig.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

        #print("\n--- Execution Complete ---")
        #print(f"A single, beautified image has been saved as: {save_path}")

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

def get_fisher_exact_value(input_df, gene_list,colname_feature,top_num,colname_Rank='Shap_abs'):
    # 2. 按照 Shap_abs 列从大到小排序
    input_df.sort_values(by=colname_Rank, ascending=False, inplace=True)

    # 3. 定义“top”基因列表，并进行统计
    # 您可以根据需要调整这个阈值，例如前20，前50等
    top_genes = set(input_df.head(top_num)[colname_feature])

    both_gene_ls = list(set(input_df[colname_feature]) & set(gene_list))
    # 4. 构建2x2列联表（Contingency Table）
    # 这个表将用于 Fisher's Exact Test
    # ----------------------------------------------------
    #               |  在Top N基因中  |  不在Top N基因中  |
    # ----------------------------------------------------
    #  在基因集中    |       a         |       b         |
    # ----------------------------------------------------
    #  不在基因集中  |       c         |       d         |
    # ----------------------------------------------------

    # a: 基因集中且在Top N中的基因数量
    in_set_and_in_top = len(top_genes.intersection(set(both_gene_ls)))

    # b: 基因集中但不在Top N中的基因数量
    in_set_and_not_in_top = len(set(both_gene_ls)) - in_set_and_in_top

    # c: 不在基因集中但在Top N中的基因数量
    not_in_set_and_in_top = len(top_genes) - in_set_and_in_top

    # d: 既不在基因集中也不在Top N中的基因数量
    not_in_set_and_not_in_top = (len(input_df) - len(top_genes)) - in_set_and_not_in_top

    # 创建列联表
    contingency_table = [[in_set_and_in_top, in_set_and_not_in_top],
                        [not_in_set_and_in_top, not_in_set_and_not_in_top]]

    #print("\n")

    # 5. 执行 Fisher's Exact Test
    # odds_ratio > 1 表示富集，p_value < 0.05 通常认为显著
    odds_ratio, p_value = fisher_exact(contingency_table)

    #print(f"Fisher's Exact Test 结果:")
    #print(f"Odds Ratio (优势比): {odds_ratio:.4f}")
    #print(f"P-value (P值): {p_value:.4e}")

    return odds_ratio, p_value


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
from typing import Dict, Any, Union
from matplotlib.lines import Line2D

# 确保中文字体已设置
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False 
except Exception:
    print("警告：中文字体配置失败。")

def plot_odds_ratio_and_log_pvalue_combined(data: pd.DataFrame, title: str, save_path: str):
    """
    将 Odds Ratio (柱状图) 和 -Log10(P-value) (折线图) 绘制在同一张图中。
    移除曲线下方的填充，使视觉更加简洁。
    """
    
    # 启用 Seaborn 风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 1. 数据预处理 ---
    plot_data = data.copy()
    plot_data = plot_data[plot_data['p_value'] > 0].reset_index(drop=True)
    
    if plot_data.empty or plot_data['odds_ratio'].isnull().all():
        print("警告：数据为空，无法绘图。")
        return
    
    log_p_values = -np.log10(plot_data['p_value'])
    SIGNIFICANCE_THRESHOLD = -np.log10(0.05) # 约 1.301
    
    # --- 2. 创建画布和主坐标轴 (Odds Ratio - 左侧) ---
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    OR_COLOR = '#1f77b4'  # 深蓝色
    PVAL_COLOR = '#d62728' # 醒目的红色
    
    # 计算柱子宽度
    if len(plot_data) > 1:
        bar_width = (plot_data['top_num'].iloc[1] - plot_data['top_num'].iloc[0]) * 0.8
    else:
        bar_width = 1
        
    # 绘制 Odds Ratio 柱状图 (左轴)
    ax1.bar(plot_data['top_num'], plot_data['odds_ratio'], 
            color=OR_COLOR, alpha=0.4, label='Odds Ratio',
            width=bar_width, zorder=1)
    
    # 绘制 OR = 1 参考线
    ax1.axhline(y=1, color='navy', linestyle='--', linewidth=1.2, alpha=0.7)
    
    # 设置左轴标签
    ax1.set_xlabel('Top Num', fontsize=12)
    ax1.set_ylabel('Odds Ratio Value', color=OR_COLOR, fontsize=12, weight='bold')
    ax1.tick_params(axis='y', labelcolor=OR_COLOR)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)

    # --- 3. 创建次坐标轴 (-Log10 P-value - 右侧) ---
    ax2 = ax1.twinx() 
    
    # 绘制 -Log10(P-value) 折线图 (右轴)
    # 注意：此处移除了 fill_between
    ax2.plot(plot_data['top_num'], log_p_values, 
             color=PVAL_COLOR, marker='o', markersize=3,
             linewidth=2, label='-Log10(P-value)', zorder=3)
             
    # 绘制 P=0.05 显著性阈值线 (右轴)
    ax2.axhline(y=SIGNIFICANCE_THRESHOLD, 
                color='darkorange', linestyle=':', 
                linewidth=2, alpha=0.9, zorder=2)
    
    # 设置右轴标签
    ax2.set_ylabel('-Log10(P-value)', color=PVAL_COLOR, fontsize=12, weight='bold')
    ax2.tick_params(axis='y', labelcolor=PVAL_COLOR)
    ax2.grid(False) # 保持右轴网格关闭，避免画面杂乱

    # --- 4. 范围自动调整 ---
    y_max_or = plot_data['odds_ratio'].max()
    ax1.set_ylim(0, y_max_or * 1.2 if np.isfinite(y_max_or) else 3)
    
    y_max_logp = log_p_values.max()
    # 确保 Y 轴高度至少能盖过显著性阈值线
    ax2.set_ylim(0, max(y_max_logp * 1.2, SIGNIFICANCE_THRESHOLD * 1.5) if np.isfinite(y_max_logp) else 5)

    # --- 5. 总体美化和图例 ---
    plt.title(title, fontsize=16, weight='bold', pad=20)
    
    # 重新构建图例元素 (移除填充框)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=OR_COLOR, alpha=0.4, label='Odds Ratio'),
        Line2D([0], [0], color=PVAL_COLOR, lw=2, marker='o', label='-Log10(P-value)'),
        Line2D([0], [0], color='navy', lw=1.2, ls='--', label='OR = 1'),
        Line2D([0], [0], color='darkorange', lw=2, ls=':', label='P = 0.05 Threshold')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()