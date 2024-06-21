# %%
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


# %%
songs = pd.read_csv("Data/songs_w_features_year.csv")
songs.drop(columns=['X', 'Artist', 'Title', 'URI', 'Release_Year'], inplace=True)
songs = pd.get_dummies(songs, columns=['Key', 'Mode', 'Time_Signature'])



# %%
# Train-Test Split
X = songs.drop(columns=['Top100'])
y = songs['Top100']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


# %%
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# #### NOT USING PCA AS IT IS NOT HELPFUL

# %%
pca = PCA(n_components = 20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(X_train.shape)



# %%

# Model 0: Baseline
baseline_model = DummyClassifier(random_state=123)
baseline_model.fit(X_train_scaled, y_train)
baseline_pred = baseline_model.predict(X_test_scaled)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_fpr, baseline_tpr, _ = roc_curve(y_test, baseline_model.predict_proba(X_test_scaled)[:,1])
baseline_auc = auc(baseline_fpr, baseline_tpr)
print("Model 0 (Baseline) Accuracy:", baseline_accuracy)
print("Model 0 (Baseline) AUC:", baseline_auc)



# %%
# Model 1: Logistic Regression
log_model = LogisticRegression(random_state=123)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, log_pred)
log_fpr, log_tpr, _ = roc_curve(y_test, log_model.predict_proba(X_test_scaled)[:,1])
log_auc = auc(log_fpr, log_tpr)

print("Model 1 (Logistic Regression) Accuracy:", log_accuracy)
print("Model 1 (Logistic Regression) AUC:", log_auc)


# %%

# Model 2: Linear Discriminant Analysis
lda_model = LDA()
lda_model.fit(X_train_scaled, y_train)
lda_pred = lda_model.predict(X_test_scaled)
lda_accuracy = accuracy_score(y_test, lda_pred)
lda_fpr, lda_tpr, _ = roc_curve(y_test, lda_model.predict_proba(X_test_scaled)[:,1])
lda_auc = auc(lda_fpr, lda_tpr)

print("Model 2 (LDA) Accuracy:", lda_accuracy)
print("Model 2 (LDA) AUC:", lda_auc)

# %%
# Model 3: Random Forest
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
rf_auc = auc(rf_fpr, rf_tpr)

print("Model 3 (Random Forest) Accuracy:", rf_accuracy)
print("Model 3 (Random Forest) AUC:", rf_auc)

# %%
# Model 4: Bagging
bag_model = BaggingClassifier(random_state=123)
bag_model.fit(X_train, y_train)
bag_pred = bag_model.predict(X_test)
bag_accuracy = accuracy_score(y_test, bag_pred)
bag_fpr, bag_tpr, _ = roc_curve(y_test, bag_model.predict_proba(X_test)[:,1])
bag_auc = auc(bag_fpr, bag_tpr)


print("Model 4 (Bagging) Accuracy:", bag_accuracy)
print("Model 4 (Bagging) AUC:", bag_auc)


# %%

#Model 5: AdaBoost
adaboost_model=AdaBoostClassifier()
adaboost_model.fit(X_train_scaled, y_train)
adaboost_pred=adaboost_model.predict(X_test_scaled)
adaboost_accuracy=accuracy_score(y_test,adaboost_pred)
adaboost_fpr, adaboost_tpr, _ = roc_curve(y_test, adaboost_model.predict_proba(X_test_scaled)[:,1])
adaboost_auc = auc(adaboost_fpr, adaboost_tpr)


print("Model 5 (AdaBoost) Accuracy:", adaboost_accuracy)
print("Model 5 (AdaBoost) AUC:", adaboost_auc)

# %%

#Model 6: Quadratic Discriminant Analysis
QDA_model=QDA()
QDA_model.fit(X_train_scaled, y_train)
QDA_pred=QDA_model.predict(X_test_scaled)
QDA_accuracy=accuracy_score(y_test,QDA_pred)
QDA_fpr, QDA_tpr, _ = roc_curve(y_test, QDA_model.predict_proba(X_test_scaled)[:,1])
QDA_auc = auc(QDA_fpr, QDA_tpr)

print("Model 6 (QDA) Accuracy:", QDA_accuracy)
print("Model 6 (QDA) AUC:", QDA_auc)

# %%

#Model 7: Gradient Boosting
GradientBoosting_model=GradientBoostingClassifier()
GradientBoosting_model.fit(X_train_scaled, y_train)
GradientBoosting_pred=GradientBoosting_model.predict(X_test_scaled)
GradientBoosting_accuracy=accuracy_score(y_test,GradientBoosting_pred)
GradientBoosting_fpr, GradientBoosting_tpr, _ = roc_curve(y_test, GradientBoosting_model.predict_proba(X_test_scaled)[:,1])
GradientBoosting_auc = auc(GradientBoosting_fpr, GradientBoosting_tpr)

print("Model 7 (Gradient Boosting) Accuracy:", GradientBoosting_accuracy)
print("Model 7 (Gradient Boosting) AUC:", GradientBoosting_auc)

# %%

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(baseline_fpr, baseline_tpr, label="Baseline (AUC = {:.2f})".format(baseline_auc))
plt.plot(log_fpr, log_tpr, label="Logistic Regression (AUC = {:.2f})".format(log_auc))
plt.plot(lda_fpr, lda_tpr, label="LDA (AUC = {:.2f})".format(lda_auc))
plt.plot(rf_fpr, rf_tpr, label="Random Forest (AUC = {:.2f})".format(rf_auc))
plt.plot(bag_fpr, bag_tpr, label="Bagging (AUC = {:.2f})".format(bag_auc))
plt.plot(adaboost_fpr, adaboost_tpr, label="Adaboost (AUC = {:.2f})".format(adaboost_auc))
plt.plot(QDA_fpr, QDA_tpr, label="QDA (AUC = {:.2f})".format(QDA_auc))
plt.plot(GradientBoosting_fpr, GradientBoosting_tpr, label="GradientBoosting (AUC = {:.2f})".format(GradientBoosting_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.savefig("images/ROC.png")
plt.legend()
plt.show()


