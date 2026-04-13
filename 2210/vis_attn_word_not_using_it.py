import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer
import os

def visualize_flux_attn(
    tensor_path, 
    prompt, 
    target_word, 
    original_image_path=None, 
    model_id=r"D:\Users\14623\Documents\Coding\ml\flux2\flux_cache\models--black-forest-labs--FLUX.2-klein-4B\snapshots\5e67da950fce4a097bc150c22958a05716994cea"
):
    """
    针对 Flux2 提取的 [1, 4096, 512] 张量进行热力图渲染
    """
    print(f"🧐 正在分析: {os.path.basename(tensor_path)}")
    
    # 1. 加载 Tokenizer 并解析 Prompt
    # 指向你本地的快照文件夹下的 tokenizer 子目录
    tokenizer_path = os.path.join(model_id, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 寻找目标词索引
    target_idx = -1
    for i, token in enumerate(tokens):
        # 处理 Byte-level BPE 特殊字符 'Ġ'
        clean_token = token.replace('Ġ', '').lower()
        if target_word.lower() == clean_token:
            target_idx = i
            print(f"✅ 找到单词 '{target_word}'，索引位置: {target_idx}")
            break
    
    if target_idx == -1:
        print(f"❌ 未能找到单词 '{target_word}'。前 20 个 Token 供参考: {tokens[:20]}")
        return

    # 2. 加载并准备张量
    if not os.path.exists(tensor_path):
        print(f"❌ 找不到张量文件: {tensor_path}")
        return
        
    attn_map = torch.load(tensor_path, map_location="cpu").squeeze(0) # -> [4096, 512]
    
    # 提取目标词对应的注意力列 [4096]
    heatmap_vec = attn_map[:, target_idx].float().numpy()

    # 3. 维度重塑 (Reshape)
    # 将 1D 序列恢复为 2D 空间图。4096 = 64x64
    grid_size = int(np.sqrt(heatmap_vec.shape[0]))
    heatmap = heatmap_vec.reshape(grid_size, grid_size)

    # 4. 归一化 (Min-Max Scaling)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 5. 绘图与渲染
    fig, ax = plt.subplots(figsize=(10, 10))

    if original_image_path and os.path.exists(original_image_path):
        print(f"🖼️ 正在叠加原图: {original_image_path}")
        img = Image.open(original_image_path).convert("RGB")
        ax.imshow(img)
        # 使用 bicubic 插值平滑马赛克，alpha 设置透明度
        ax.imshow(heatmap, cmap='magma', alpha=0.6, extent=(0, img.size[0], img.size[1], 0), interpolation='bicubic')
    else:
        # 只显示热力图
        print("💡 未提供原图，仅生成热力图。")
        im = ax.imshow(heatmap, cmap='magma', interpolation='bicubic')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Intensity')
    
    plt.title(f"Flux Attention Map: '{target_word}'\n(Step/Layer Info from Filename)", fontsize=14)
    plt.axis('off')
    
    # 强制将图片保存在脚本所在的文件夹下
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_base_name = os.path.basename(tensor_path).replace(".pt", "")
    output_filename = f"vis_{target_word}_{file_base_name}.png"
    save_path = os.path.join(current_script_dir, output_filename)
    
    # 保存并关闭
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
    
    print(f"🚀 可视化完成！图片已保存至: {save_path}")

# --- 执行入口 ---
if __name__ == "__main__":
    PROMPT = "better resolution, photo-realistic, a cat saying hello world" 
    TENSOR_FILE = r".\2210\flux_cross_attn_map_optimized.pt"
    ORIGINAL_IMG = r".\2210\flux-klein.png" 
    
    visualize_flux_attn(
        TENSOR_FILE, 
        PROMPT, 
        target_word="cat", 
        original_image_path=ORIGINAL_IMG
    )