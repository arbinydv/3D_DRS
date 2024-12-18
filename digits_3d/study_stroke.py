import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the data without headers
file_path = 'training_data/stroke_9_0052.csv'
data = pd.read_csv(file_path, names=['x', 'y', 'z'])  # Add column names explicitly

# Step 2: Extract x, y, z coordinates
x = data['x']
y = data['y']
z = data['z']

# Step 3: Create the figure with subplots
fig = plt.figure(figsize=(12, 6))  # Adjust the overall figure size

# Subplot 1: 3D Line Plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # 1 row, 2 columns, position 1
ax1.plot(x, y, z, color='blue', linewidth=2)
ax1.set_title('3D Line Plot')
ax1.set_xlabel('X Cords')
ax1.set_ylabel('Y Cords')
ax1.set_zlabel('Z Cords')

# Subplot 2: 2D Line Plot (X-Y Plane)
ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, position 2
ax2.plot(x, y, color='red', linewidth=2)
ax2.set_title('2D Plot (X-Y Plane)')
ax2.set_xlabel('X Cords')
ax2.set_ylabel('Y Cords')

# Show the combined plots
plt.tight_layout()
plt.show()