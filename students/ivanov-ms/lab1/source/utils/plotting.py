import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import roc_curve, roc_auc

IMAGES_DIR = "./images/"


def _save_and_close(img_name: str):
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)
        print(f"Created directory for images: {os.path.abspath(IMAGES_DIR)}")

    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close('all')


def plot_roc_curve(y_true, y_scores_dict: dict, img_name="roc_curve.png"):
    plt.figure(figsize=(10, 8))
    
    for label, y_scores in y_scores_dict.items():
        fpr, tpr = roc_curve(y_true, y_scores)
        auc_score = roc_auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=label + f" (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    _save_and_close(img_name)


def plot_confusion_matrix(cm, title="Confusion Matrix", img_name="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=cm.columns.get_level_values(1), 
                yticklabels=cm.index.get_level_values(1))
    plt.ylabel(cm.index.name)
    plt.xlabel(cm.columns.name)
    plt.title(title)
    _save_and_close(img_name)


def plot_feature_importances(feat_imp, feat_names=None, img_name="feature_importances.png"):
    imp_df = pd.DataFrame(data={"Importance": feat_imp}, index=feat_names)
    imp_df.sort_values("Importance", ascending=False, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    imp_df.plot.bar(rot=90, ax=ax)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.set_title("Features Importance")
    _save_and_close(img_name)
