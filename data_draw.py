import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
file_path_1 = '/Users/xiaoxizhou/Downloads/adrian_surf/code/pinn_predictions.csv'
file_path_2 = '/Users/xiaoxizhou/Downloads/adrian_surf/code/training_data.csv'
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Define columns
x_col = 'log10_t'
y_cols = ['X_CO2', 'X_O2', 'X_N2', 'X_CO', 'X_NO', 'X_C', 'X_O', 'X_N']

# Create the plot
plt.figure(figsize=(12, 7))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, col in enumerate(y_cols):
    # 为每个物种分配一个颜色
    current_color = colors[i % len(colors)] 
    
    # 绘制预测值（虚线）
    plt.plot(df1[x_col], df1[col], linestyle='--', color=current_color, label=f'Pred {col}')
    
    # 绘制真实值（实线，增加透明度以防遮挡）
    plt.plot(df2[x_col], df2[col], color=current_color, alpha=0.4, label=f'True {col}')


# Formatting the plot
plt.xlabel(r'$\log_{10}(t)$')
plt.ylabel('Mole Fraction')
plt.yscale('log')  # Log scale for y-axis to see all species
plt.title('PINN: MLP + physics loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(False)
plt.tight_layout()

# Save and display the result
plt.show()
