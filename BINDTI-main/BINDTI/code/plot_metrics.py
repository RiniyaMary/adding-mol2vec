import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch
# Load predictions
df = pd.read_csv("../output/visualization.csv")
y_true = df["y_label"]
y_pred = df["y_pred"]

# -------- ROC Curve --------
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("../output/roc_curve.png")
plt.close()

# -------- PR Curve --------
precision, recall, _ = precision_recall_curve(y_true, y_pred)
pr_auc = average_precision_score(y_true, y_pred)

plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("../output/pr_curve.png")
plt.close()

print("✅ ROC & PR curves saved")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Choose threshold (use 0.5 or your best threshold)
threshold = 0.5
y_bin = (y_pred >= threshold).astype(int)

cm = confusion_matrix(y_true, y_bin)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("../output/confusion_matrix.png")
plt.close()

print("✅ Confusion matrix saved")



metrics = torch.load(r"C:\Users\Rony Pradeep\OneDrive\Documents\mol2vec\BINDTI-main\BINDTI\output\result\sample\random\result_metrics.pt")
val_auroc = metrics["val_auroc_epoch"]

plt.figure()
plt.plot(range(1, len(val_auroc)+1), val_auroc)
plt.xlabel("Epoch")
plt.ylabel("Validation AUROC")
plt.title("AUROC vs Epoch")
plt.savefig("../output/auroc_per_epoch.png")
plt.close()

print("✅ AUROC per epoch saved")

