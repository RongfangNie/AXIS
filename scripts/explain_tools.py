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