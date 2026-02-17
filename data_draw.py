import pandas as pd
import matplotlib.pyplot as plt

# Load the training data
file_path_1 = '/Users/xiaoxizhou/Downloads/adrian_surf/code/data_predictions.csv'
file_path_2 = '/Users/xiaoxizhou/Downloads/adrian_surf/code/training_data.csv'
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Define columns
x_col = 'log10_t'
y_cols = ['X_CO2', 'X_O2', 'X_N2', 'X_CO', 'X_NO', 'X_C', 'X_O', 'X_N']

# Create the plot
plt.figure(figsize=(10, 6))

for col in y_cols:
    plt.plot(df1[x_col], df1[col], label=f'Prediction {col}')
    plt.plot(df2[x_col], df2[col], label=f'Training Data {col}', alpha=0.5)

# Formatting the plot
plt.xlabel(r'$\log_{10}(t)$')
plt.ylabel('Mole Fraction')
plt.yscale('log')  # Log scale for y-axis to see all species
plt.title('Species Concentration vs Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which="both", linestyle="--", alpha=0.2)
plt.tight_layout()

# Save and display the result
plt.show()
