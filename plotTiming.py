import matplotlib.pyplot as plt
import numpy as np

# Path to the text file
file_path = 'timing.txt'

# Read data from the file
data = np.loadtxt(file_path, dtype=int)

# Assuming each column is a node and each row is a frame
# Split the data into four separate arrays for plotting
node1 = data[:, 0]  # First column - Node 1
node2 = data[:, 1]  # Second column - Node 2
node3 = data[:, 2]  # Third column - Node 3
node4 = data[:, 3]  # Fourth column - Node 4

# Plotting
plt.figure(figsize=(6, 4))

plt.plot(node1, label=f'ros2im mu = {np.mean(node1):.2f}', color='red')
plt.plot(node2, label=f'im2yolo mu = {np.mean(node2):.2f}', color='blue')
plt.plot(node3, label=f'yolo2state mu = {np.mean(node3):.2f}', color='green')
plt.plot(node4, label=f'state2disp mu = {np.mean(node4):.2f}', color='black')

plt.title('YoloV8l SingleCamera Delay')
plt.xlabel('Processed Frame')
plt.ylabel('Time(ms)')
plt.legend()

#plot_filename = file_path.replace('.txt', '.png')

# Save the plot
#plt.savefig(plot_filename)

# Optionally, display the plot
plt.show()