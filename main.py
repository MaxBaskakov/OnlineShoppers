import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


data = pd.read_csv("online_shoppers_intention.csv")
data.info()

data["Weekend"] = data["Weekend"].map({True: 1, False: 0})
data_train, data_val = train_test_split(data, test_size=0.4, random_state=42)
data_val, data_test = train_test_split(data_val, test_size=0.5, random_state=42)

input_cols = data.drop(columns="Revenue").columns.tolist().copy()
target_cols = "Revenue"

target_train = data_train[target_cols]
val_target = data_val[target_cols]
test_target = data_test[target_cols]

numeric_cols = data_train[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = data_train[input_cols].select_dtypes(include=str).columns.tolist()

scaler = MinMaxScaler().fit(data_train[numeric_cols])

data_train[numeric_cols] = scaler.transform(data_train[numeric_cols])
data_val[numeric_cols] = scaler.transform(data_val[numeric_cols])
data_test[numeric_cols] = scaler.transform(data_test[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(data_train[categorical_cols])
encoded_columns = encoder.get_feature_names_out(categorical_cols).tolist()

data_train[encoded_columns] = encoder.transform(data_train[categorical_cols])
data_val[encoded_columns] = encoder.transform(data_val[categorical_cols])
data_test[encoded_columns] = encoder.transform(data_test[categorical_cols])

x_train = data_train[numeric_cols+encoded_columns]
x_val = data_val[numeric_cols+encoded_columns]
x_test = data_test[numeric_cols+encoded_columns]

rf = RandomForestClassifier(random_state=42).fit(x_train,target_train)
predict_rf_val = rf.predict(x_val)
acc_rf = accuracy_score(val_target, predict_rf_val)

xgb = XGBClassifier(random_state=42, ).fit(x_train,target_train)
predict_xgb_val = xgb.predict(x_val)
acc_xgb = accuracy_score(val_target, predict_xgb_val)


print(f"Untuned accuracy random forest{acc_rf}, Untuned accuracy xgb{acc_xgb}")

###RF
acc_rf= accuracy_score(val_target, predict_rf_val)
precision_rf = precision_score(val_target, predict_rf_val)
recall_rf = recall_score(val_target, predict_rf_val)
f1_rf = f1_score(val_target, predict_rf_val)
roc_fr = roc_auc_score(val_target, rf.predict_proba(x_val)[:, 1])

###XGB
acc_xgb = accuracy_score(val_target, predict_xgb_val)
precision_xgb = precision_score(val_target, predict_xgb_val)
recall_xgb = recall_score(val_target, predict_xgb_val)
f1_xgb = f1_score(val_target, predict_xgb_val)
roc_xgb = roc_auc_score(val_target, xgb.predict_proba(x_val)[:, 1])

result_metrics1 = pd.DataFrame({
    "Models": ["RandomForest","XGBClassifier"],
    "Accuracy": [acc_rf, acc_xgb],
    "Precison":[precision_rf, precision_xgb],
    "Recall": [recall_rf, recall_xgb],
    "F1": [f1_rf, f1_xgb],
    "ROC-AUC": [roc_fr,roc_xgb]

})

print(result_metrics1)

# hyperparameters tuning
rf = RandomForestClassifier(random_state=42)

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}
rf_grid = GridSearchCV(rf,rf_params,cv=5,scoring="f1", n_jobs=-1)
rf_grid.fit(x_train,target_train)
print(f"\nBest params for RandomForest: {rf_grid.best_params_}")
best_rf = rf_grid.best_estimator_
best_rf_pred = best_rf.predict(x_val)

tuned_acc_rf= accuracy_score(val_target, best_rf_pred)
tuned_precision_rf = precision_score(val_target, best_rf_pred)
tuned_recall_rf = recall_score(val_target, best_rf_pred)
tuned_f1_rf = f1_score(val_target, best_rf_pred)
tuned_roc_fr = roc_auc_score(val_target, best_rf.predict_proba(x_val)[:, 1])

with open("best_rf_params.json", "w") as f:
    json.dump(rf_grid.best_params_, f, indent=4)

xgb = XGBClassifier(random_state=42)

xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb, xgb_params, cv=5,scoring="f1", n_jobs=-1)
xgb_grid.fit(x_train,target_train)
print(f"\nBest params for xgb: {xgb_grid.best_params_}")
best_xgb = xgb_grid.best_estimator_
best_xgb_predict = best_xgb.predict(x_val)

tuned_acc_xgb = accuracy_score(val_target, best_xgb_predict)
tuned_precision_xgb = precision_score(val_target, best_xgb_predict)
tuned_recall_xgb = recall_score(val_target, best_xgb_predict)
tuned_f1_xgb = f1_score(val_target, best_xgb_predict)
tuned_roc_xgb = roc_auc_score(val_target, best_xgb.predict_proba(x_val)[:, 1])

with open("best_xgb_params.json", "w") as f:
    json.dump(xgb_grid.best_params_, f, indent=4)

result_tuned = pd.DataFrame({
    "TunedModel": ["RandomForest", "XGBClassifier"],
    "Accuracy": [tuned_acc_rf, tuned_acc_xgb],
    "Precison":[tuned_precision_rf, tuned_precision_xgb],
    "Recall": [tuned_recall_rf, tuned_recall_xgb],
    "F1": [tuned_f1_rf, tuned_f1_xgb],
    "ROC-AUC": [tuned_roc_fr,tuned_roc_xgb]
})
print(f"\n{result_tuned}")

ConfusionMatrixDisplay.from_predictions(val_target, best_xgb_predict)
plt.title("XGBoost Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.show()

feature_importance = pd.Series(best_xgb.feature_importances_,index=x_train.columns).sort_values(ascending=False)

feature_importance.head(15).sort_values().plot(kind="barh", figsize=(10, 6))

plt.title("Top 15 XGBoost Feature Importances")
plt.xlabel("Importance")
plt.savefig("plots/FeatureImportances.png")
plt.show()

comp_basic_tuned = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    "Basic XGB": [acc_xgb, precision_xgb, recall_xgb, f1_xgb,roc_xgb],
    "Tuned XGB": [tuned_acc_xgb, tuned_precision_xgb, tuned_recall_xgb, tuned_f1_xgb, tuned_roc_xgb]
})

comp_basic_tuned.plot(x="Metric", kind="bar", figsize=(10, 6))

plt.title("XGBoost:Basic vs Tuned")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.savefig("plots/BasicVSTuned.png")
plt.show()

model = best_xgb
predict_test = model.predict(x_test)

test_accuracy = accuracy_score(test_target, predict_test)
test_precision = precision_score(test_target, predict_test)
test_recall = recall_score(test_target, predict_test)
test_f1 = f1_score(test_target, predict_test)
test_roc_auc = roc_auc_score(test_target, model.predict_proba(x_test)[:, 1])

print("Final results on test data")
print(f"Accuracy:{test_accuracy:.4f}"
      f"\nPrecision:{test_precision:.4f}"
      f"\nRecall:{test_recall:.4f}"
      f"\nF1:{test_f1:.4f}"
      f"\nROC-AUC:{test_roc_auc:.4f}")

def predict(data):
        data = data.copy()
        data["Weekend"] = data["Weekend"].map({True:1,False:0})
        data[numeric_cols] = scaler.transform(data[numeric_cols])
        data[encoded_columns] = encoder.transform(data[categorical_cols])
        data_x = data[numeric_cols+encoded_columns]
        prediction = model.predict(data_x)
        probability = model.predict_proba(data_x)[0, 1]
        if prediction[0] == 1:
            return "Customer", probability
        else:
            return "Not a Customer", probability

new_data = pd.DataFrame([{
    "Administrative": 2,
    "Administrative_Duration": 50.0,
    "Informational": 1,
    "Informational_Duration": 20.0,
    "ProductRelated": 20,
    "ProductRelated_Duration": 500.0,
    "BounceRates": 0.02,
    "ExitRates": 0.03,
    "PageValues": 10.0,
    "SpecialDay": 0.0,
    "Month": "May",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}])

result, probability = predict(new_data)

print(f"\nResult for new data: {result}")
print(f"Probability: {probability:.2%}")