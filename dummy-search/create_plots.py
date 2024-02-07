from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from clip import Clip
from app import load_all_images, rank_images, save_plot


def create_multi_roc_curve(data: dict):
    """
    Plot ROC curves for multiple queries.

    :param data: A dictionary where keys are query names and values are lists of Image objects.
    """
    plt.figure(figsize=(10, 8))

    for query, images in data.items():
        step_size = 10
        thresholds = [t + step_size for t in range(0, len(images), step_size)]
        y = [0]  # tpr
        x = [0]  # fpr

        positives = sum(1 for image in images if image.is_relevant)
        negatives = sum(1 for image in images if not image.is_relevant)

        for t in thresholds:
            true_positives = 0
            false_positives = 0

            for image in images[:t]:
                if image.is_relevant:
                    true_positives += 1
                else:
                    false_positives += 1

            tpr = true_positives / positives if positives else 0
            fpr = false_positives / negatives if negatives else 0

            y.append(tpr)
            x.append(fpr)

        roc_auc = auc(x, y)

        # Plot the ROC curve for the current query
        plt.plot(x, y, lw=2, label=f"{query} (area = {roc_auc:.2f})")

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_plot(plt, "roc_combined", "-".join(data.keys()))


if __name__ == "__main__":
    load_dotenv()

    print("Loading CLIP...")
    clip = Clip()
    print("Done")

    print("Loading images...")
    images = load_all_images()
    print("Done")

    queries = ["basketball", "baseball", "baseball", "football", "car", "native"]
    data = {}

    for q in queries:
        data[q] = rank_images(images, q, "Cosine Similarity", clip)

    create_multi_roc_curve(data)
