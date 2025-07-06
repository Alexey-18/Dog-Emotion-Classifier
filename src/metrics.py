# Evaluation - Comprehensive Metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from emotion import train_gen, val_gen, model
import json

#Color palette
PALETTE = ['#A0C4FF', '#BDB2FF', '#FFC6FF', '#FFADAD', '#CAFFBF', '#FDFFB6']

def comprehensive_evaluation(model, val_gen, class_names):
    """Comprehensive model evaluation with multiple metrics and enha nced visualizations"""

    print("\n Comprehensive Model Evaluation")
    print("-" * 20)

    #Get predictions
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    #Basic metrics
    accuracy = np.mean(y_pred == y_true)
    print(f"Overall Accuracy: {accuracy:.4f}")

    #Detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    #Set global style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '0.4',
        'axes.labelcolor': '0.2',
        'axes.titleweight': 'bold',
        'grid.color': '0.92',
        'xtick.color': '0.4',
        'ytick.color': '0.4',
    })

    #Create custom colormaps
    cmap1 = sns.light_palette(PALETTE[0], as_cmap=True)
    cmap2 = sns.light_palette(PALETTE[1], as_cmap=True)

    #1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap1, xticklabels=class_names, yticklabels=class_names, cbar_kws={'shrink': 0.75})
    plt.title('Confusion Matrix', fontsize=18, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha ='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    #2. Mormalizer Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap2, xticklabels=class_names, yticklabels=class_names, cbar_kws={'shrink': 0.75})
    plt.title('Normalizer Confusion Matrix', fontsize=18, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    #3. Per-class metrics
    metrics_data = []
    for i, class_name in enumerate(class_names):
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        metrics_data.append([class_name, precision, recall, f1, support])

    metrics_df = pd.DataFrame(metrics_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

    plt.figure(figsize=(12, 7))
    x_pos = np.arange(len(class_names))
    width = 0.25
    bar_thickness = 0.8

    plt.bar(x_pos - width, metrics_df['Precision'], width, label='Precision', alpha=0.9, color=PALETTE[0], edgecolor='white', linewidth=1)
    plt.bar(x_pos, metrics_df['Recall'], width, label='Recall', alpha=0.9, color=PALETTE[1], edgecolor='white', linewidth=1)
    plt.bar(x_pos + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.9, color=PALETTE[2], edgecolor='white', linewidth=1)

    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Per-Class Metrics', fontsize=18, pad=15)
    plt.xticks(x_pos, class_names, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.15)
    plt.legend(frameon=True, framealpha=0.9, loc='upper right')
    plt.grid(True, axis='y', alpha=0.4)

    #Add data labels
    for i, vals in enumerate(zip(metrics_df['Precision'], metrics_df['Recall'], metrics_df['F1-Score'])):
        for j, val in enumerate(vals):
            offset = (j - 1) * width
            plt.text(i + offset, val + 0.02, f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.show()

    #4. ROC Curves (One-vs-Rest)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        auc_score = roc_auc_score(y_true_binary,  y_pred_proba[:, i])
        plt.plot(fpr, tpr, lw=2.5, label=f'{class_name} (AUC = {auc_score:.3f})',color=PALETTE[i % len(PALETTE)])
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=18, pad=15)
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #5. Precision_Recall Curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
        ap_score = average_precision_score(y_true_binary, y_pred_proba[:, i])
        plt.plot(recall_curve, precision_curve, lw=2.5, label=f'{class_name} (AP = {ap_score:.3f})', color=PALETTE[i % len(PALETTE)])

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precisions-Recall Curves', fontsize=18, pad=15)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #6. Prediction Confidence Distribution
    plt.figure(figsize=(10, 6))
    confidence_scores = np.max(y_pred_proba, axis=1)
    correct_predictions = (y_pred == y_true)

    plt.hist(confidence_scores[correct_predictions], bins=20, alpha=0.85, label='Correct Predictions', color= PALETTE[4], edgecolor='white')
    plt.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.85, label='Incorrect Predictions', color=PALETTE[3], edgecolor='white')
    plt.xlabel('Prediction Confidence', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Prediction Confidence Distribution', fontsize=18, pad=15)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Print detailed metrics
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    #Calculate and print additional metrics
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    print(f"\nSummary Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    #Per-class AUC scores
    print(f"\nPer-Class AUC Scores:")
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        auc_score = roc_auc_score(y_true_binary, y_pred_proba[:, i])
        print(f"{class_name}: {auc_score:.4f}")
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'true_labels': y_true
    }
#Get class names in correct order
class_names = [k for k, v in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]

#Evaluate model 
eval_results = comprehensive_evaluation(model, val_gen, class_names)

#Save the entire model
model.save('dog_emotion_classifier.h5')
print(" Model saved as 'dog_emotion_classifier.h5'")

#Save class indices
with open('model/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)
print("  Class indices saved as 'class_indices.json'")
