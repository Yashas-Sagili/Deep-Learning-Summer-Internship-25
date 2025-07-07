from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Normal", "Covid"], yticklabels=["Normal", "Covid"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
