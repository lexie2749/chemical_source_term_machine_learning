import matplotlib.pyplot as plt

def draw_mlp(layers):
    """
    绘制多层感知器 (MLP) 架构图。
    :param layers: 包含各层神经元数量的列表。
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # 水平间距
    h_spacing = 1.0 / (len(layers) + 1)
    node_coords = []

    # 1. 计算坐标并绘制神经元
    for i, layer_size in enumerate(layers):
        layer_coords = []
        x = (i + 1) * h_spacing
        
        for j in range(layer_size):
            y = (j + 1) * (1.0 / (layer_size + 1))
            layer_coords.append((x, y))
            
            # 绘制神经元 (Circle)
            circle = plt.Circle((x, y), 0.015, color='#3498db', ec='black', zorder=4)
            ax.add_artist(circle)
            
        node_coords.append(layer_coords)

    # 2. 绘制层与层之间的连接线 (Synapses)
    for i in range(len(node_coords) - 1):
        for start_node in node_coords[i]:
            for end_node in node_coords[i+1]:
                line = plt.Line2D([start_node[0], end_node[0]], 
                                  [start_node[1], end_node[1]], 
                                  color='gray', linewidth=0.1, alpha=0.3, zorder=1)
                ax.add_artist(line)

    # 标注
    plt.title(f"MLP Architecture: {'-'.join(map(str, layers))}", fontsize=15)
    plt.text(h_spacing, -0.02, "Input Layer", ha='center', fontsize=12)
    plt.text(len(layers) * h_spacing, -0.02, "Output Layer", ha='center', fontsize=12)
    
    # 标注隐藏层
    for i in range(1, len(layers) - 1):
        plt.text((i + 1) * h_spacing, -0.02, f"Hidden {i}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('mlp_architecture.png', dpi=300, bbox_inches='tight')

# 绘图调用
draw_mlp([1, 32, 32, 32, 10])