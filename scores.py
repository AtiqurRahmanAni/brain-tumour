import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score, roc_curve, roc_auc_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt

def score_in_details(pred, real, probas):
  class_name = [0, 1]
  y_pred = pred
  y_real = real
    
  accuracy = recall_score(y_real, y_pred, average='weighted')
  precision = precision_score(y_real, y_pred, average='weighted')
  recall = recall_score(y_real, y_pred, average='weighted')
  f1 = f1_score(y_real, y_pred, average='weighted')
  roc_auc = roc_auc_score(y_real, probas, average='weighted')
  mcc_score = matthews_corrcoef(y_real, y_pred)
  fpr, tpr, _ = roc_curve(y_real, probas)

  print(f"Accuracy: {accuracy * 100}%")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")
  print(f"MCC-score: {mcc_score}")
  print(f"ROC AUC score: {roc_auc}")
  
  print()
  print()


  _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=1000)
  conf_matrix = confusion_matrix(y_real, y_pred)
  sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap ='Blues', fmt='g', xticklabels=class_name, yticklabels=class_name, ax=ax1)
  ax1.set_title('Confusion matrix')
  ax1.set_ylabel('Actual label')
  ax1.set_xlabel('Predicted label')

  ax2.plot(fpr, tpr)
  ax2.plot([0,1], [0,1], linestyle='--')
  ax2.set_ylabel('TPR')
  ax2.set_xlabel('FPR')
  ax2.set_title('ROC curve')

  plt.show()