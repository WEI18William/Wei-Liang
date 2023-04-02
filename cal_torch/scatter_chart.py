import matplotlib.pyplot as plt
import os

def plot_scatter_3d(centers_dict,class_names,class_labels,output_path):

    # Create figure and axes objects
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop through class labels and plot centers for each class
    for i, label in enumerate(class_labels):
        centers = centers_dict[i]
        x_vals = [center[2] for center in centers] # x-axis values
        y_vals = [center[1] for center in centers] # y-axis values
        z_vals = [center[0] for center in centers] # z-axis values
        ax.scatter(x_vals, y_vals, z_vals, label=class_names[i])

    # Add legend and axis labels
    ax.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show plot
    plt.savefig(os.path.join(output_path))

