import torch
from scipy.spatial.distance import cdist

def calculate_distances(centers_dict,class_names):
    class_labels = list(centers_dict.keys())
    num_classes = len(class_labels)
    distances = torch.zeros((num_classes, num_classes))

    for i, label1 in enumerate(class_labels):
        centers1 = torch.stack([torch.tensor(center) for center in centers_dict[label1]])
        for j, label2 in enumerate(class_labels):
            # Exclude diagonal elements (self-distance)
            if i != j:
                centers2 = torch.stack([torch.tensor(center) for center in centers_dict[label2]])
                distances_ij = cdist(centers1.numpy(), centers2.numpy()).mean()
                distances[i, j] = distances_ij

    # Print distances between selected classes
    print("\nDistances between classes:")
    for i, label1 in enumerate(class_labels):
        for j, label2 in enumerate(class_labels):
            if i < j:
                print(f"{class_names[i]} to {class_names[j]}: {distances[i][j]}")


    return distances.tolist()


