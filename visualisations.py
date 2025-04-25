import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.manifold import TSNE
from collections import Counter
import torch

def plot_curve(data, x_label, y_label, title):
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def plot_top_symmetric_misclassifications(all_targets, all_predictions, top_n=10):
    """
    Identifies the top N *symmetric* misclassifications and plots them on a heatmap.
    A symmetric pair (A, B) is treated the same as (B, A) and their counts are summed.
    """
    misclassified = []
    for true, pred in zip(all_targets, all_predictions):
        if true != pred:
            # Sort the pair so (A, B) and (B, A) are treated the same
            ordered_pair = tuple(sorted((true, pred)))
            misclassified.append(ordered_pair)

    if not misclassified:
        print("No misclassifications found.")
        return

    misclass_count = Counter(misclassified)

    actual_top_n = min(top_n, len(misclass_count))
    if actual_top_n == 0:
        print("No misclassifications found.")
        return

    top_misclassifications = misclass_count.most_common(actual_top_n)

    # Extract unique labels from top pairs
    involved_labels = set()
    for a, b in [pair for pair, _ in top_misclassifications]:
        involved_labels.update([a, b])
    involved_labels = sorted(list(involved_labels))

    # Create index mapping
    label_to_idx = {label: i for i, label in enumerate(involved_labels)}
    size = len(involved_labels)
    conf_matrix = np.zeros((size, size), dtype=int)

    for (label1, label2), count in top_misclassifications:
        i = label_to_idx[label1]
        j = label_to_idx[label2]
        # Since we're making it symmetric, add the count to both (i, j) and (j, i)
        conf_matrix[i, j] += count
        conf_matrix[j, i] += count

    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=involved_labels, yticklabels=involved_labels)
    plt.xlabel("Label")
    plt.ylabel("Label")
    plt.title(f"Top {actual_top_n} Symmetric Misclassifications")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_large_confusion_matrix(conf_matrix, class_labels, figsize=(25, 20), annot_kws={"size": 8}):
    """
    Plots a large confusion matrix with adjusted figure and font sizes.
    (Assumes this function is defined as in the previous response)
    """
    if conf_matrix.shape[0] != len(class_labels) or conf_matrix.shape[1] != len(class_labels):
        raise ValueError("Matrix dimensions must match the number of class labels.")

    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                annot_kws=annot_kws)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Full Confusion Matrix")
    plt.xticks(rotation=90, fontsize=annot_kws.get("size", 8) + 1)
    plt.yticks(rotation=0, fontsize=annot_kws.get("size", 8) + 1)
    plt.tight_layout(pad=1.5)

    try:
        plt.savefig("full_confusion_matrix.pdf", format="pdf", bbox_inches='tight')
        plt.savefig("full_confusion_matrix.png", format="png", dpi=300, bbox_inches='tight')
        print("Saved matrix plots to full_confusion_matrix.pdf and .png")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

def generate_and_plot_confusion_matrix(all_targets, all_predictions, **plot_kwargs):
    """
    Calculates and plots the full confusion matrix directly from label lists.

    Args:
        all_targets (list or np.array): List of true ground truth labels.
        all_predictions (list or np.array): List of predicted labels.
        **plot_kwargs: Keyword arguments to be passed directly to the
                       plot_large_confusion_matrix function
                       (e.g., figsize=(30,25), annot_kws={"size": 7}).
    """
    # Find unique labels and sort them
    # Use np.unique on the combined list to ensure all labels are captured
    labels = sorted(list(np.unique(list(all_targets) + list(all_predictions))))
    num_labels = len(labels)
    print(f"Found {num_labels} unique classes.")

    if num_labels == 0:
        print("No labels found. Cannot generate confusion matrix.")
        return

    # Calculate the confusion matrix
    # Pass 'labels=labels' to ensure the matrix order matches our sorted list
    cm = confusion_matrix(all_targets, all_predictions, labels=labels)

    # Plot the matrix using the existing plotting function
    print("Generating confusion matrix plot...")
    plot_large_confusion_matrix(cm, labels, **plot_kwargs)

def plot_tsne(vectors, labels, selected_labels=None, max_per_label=100, title="t-SNE of Utterance Representations"):
    """
    vectors: list of torch.Tensor [n_samples, dim]
    labels: list of int or str labels, same length as vectors
    selected_labels: list of labels to include (optional)
    max_per_label: max examples to include per label
    title: plot title
    """
    # Convert tensors to a single numpy array
    vecs = torch.stack(vectors).cpu().numpy()
    labels = np.array(labels)

    # Subset by selected_labels if provided
    if selected_labels is not None:
        indices = [i for i, label in enumerate(labels) if label in selected_labels]
        vecs = vecs[indices]
        labels = labels[indices]

    # Further limit to max_per_label per class
    filtered_vecs, filtered_labels = [], []
    label_counter = Counter()

    for vec, label in zip(vecs, labels):
        if label_counter[label] < max_per_label:
            filtered_vecs.append(vec)
            filtered_labels.append(label)
            label_counter[label] += 1

    # Convert list to 2D NumPy array
    filtered_vecs = np.stack(filtered_vecs)

    # Run t-SNE
    vec_2d = TSNE(n_components=2).fit_transform(filtered_vecs)

    # Plot
    plt.figure(figsize=(12, 10))
    for label in set(filtered_labels):
        idxs = [i for i, l in enumerate(filtered_labels) if l == label]
        plt.scatter(vec_2d[idxs, 0], vec_2d[idxs, 1], label=label, alpha=0.6)

    plt.legend()
    plt.title(title)

    try:
        plt.savefig(f"{title}.png", format="png", dpi=300, bbox_inches='tight')
        print(f"Saved plots to {title}.png")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

def plot_tsne_zoomed(vectors, labels, texts, selected_labels=None, 
                    max_per_label=100, title="t-SNE Zoomed", 
                    zoom_scale=1.0, annotate=True, annotate_sample=0.5):
    """
    vectors: list of torch.Tensor [n_samples, dim]
    labels: list of int or str labels, same length as vectors
    texts: list of utterance texts, same length as vectors
    selected_labels: list of labels to include (optional)
    max_per_label: max examples to include per label
    title: plot title
    zoom_scale: how much to zoom in (1.0 = no zoom)
    annotate: whether to show text annotations
    annotate_sample: fraction of points to annotate (0-1)
    """
    # Convert tensors to a single numpy array
    vecs = torch.stack(vectors).cpu().numpy()
    labels = np.array(labels)
    texts = np.array(texts)

    # Subset by selected_labels if provided
    if selected_labels is not None:
        indices = [i for i, label in enumerate(labels) if label in selected_labels]
        vecs = vecs[indices]
        labels = labels[indices]
        texts = texts[indices]

    # Further limit to max_per_label per class
    filtered_vecs, filtered_labels, filtered_texts = [], [], []
    label_counter = Counter()

    for vec, label, text in zip(vecs, labels, texts):
        if label_counter[label] < max_per_label:
            filtered_vecs.append(vec)
            filtered_labels.append(label)
            filtered_texts.append(text)
            label_counter[label] += 1

    # Convert list to 2D NumPy array
    filtered_vecs = np.stack(filtered_vecs)

    # Run t-SNE
    vec_2d = TSNE(n_components=2).fit_transform(filtered_vecs)

    # Plot
    plt.figure(figsize=(12, 10))
    
    # First pass: plot all points
    for label in set(filtered_labels):
        idxs = [i for i, l in enumerate(filtered_labels) if l == label]
        plt.scatter(vec_2d[idxs, 0], vec_2d[idxs, 1], label=label, alpha=0.6)
    
    # Second pass: add annotations
    if annotate:
        rng = np.random.RandomState(0)  # for reproducible sampling
        for i, (x, y, label, text) in enumerate(zip(vec_2d[:, 0], vec_2d[:, 1], filtered_labels, filtered_texts)):
            # Only annotate a fraction of points for readability
            if rng.rand() < annotate_sample:
                plt.annotate(
                    f"{label}: {text[:30]}{'...' if len(text) > 30 else ''}", 
                    xy=(x, y), 
                    xytext=(5, 2), 
                    textcoords='offset points',
                    ha='left', 
                    va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Adjust zoom if needed
    if zoom_scale != 1.0:
        x_min, x_max = vec_2d[:, 0].min(), vec_2d[:, 0].max()
        y_min, y_max = vec_2d[:, 1].min(), vec_2d[:, 1].max()
        x_mid, y_mid = (x_min + x_max)/2, (y_min + y_max)/2
        x_range, y_range = (x_max - x_min)/zoom_scale, (y_max - y_min)/zoom_scale
        plt.xlim(x_mid - x_range/2, x_mid + x_range/2)
        plt.ylim(y_mid - y_range/2, y_mid + y_range/2)
    
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    try:
        plt.savefig(f"{title}.png", format="png", dpi=300, bbox_inches='tight')
        print(f"Saved plots to {title}.png")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()