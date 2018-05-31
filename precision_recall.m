function [precision, recall, F_score] = precision_recall(X_cv,y_cv,threshold,theta,rowct,colct);
  CM = confusion_matrix(X_cv,y_cv,threshold,theta,rowct,colct);
  precision = CM(1,1) / (CM(1,1) + CM(2,1));
  recall = CM(1,1) / (CM(1,1) + CM(1,2));
  F_score = 2 * ((precision * recall) / (precision + recall));
end