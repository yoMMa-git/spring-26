# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

from loosify_dataset import inject_missing_values
from impute_proba import probabilistic_impute
from tree import ID3Tree


# %%
df = pd.read_csv("../data/diabetes_binary_health_indicators_BRFSS2015.csv")

df.head()

# %%
df_missing = inject_missing_values(
    df,
    missing_ratio=0.1,
    random_state=42
)

df_missing.head()

# %%
target = "Diabetes_binary"

X_df = df_missing.drop(columns=[target])
y_df = df_missing[target]

X_filled = probabilistic_impute(X_df, random_state=42)

df_clean = pd.concat([X_filled, y_df], axis=1)

# удаляем строки где target = NaN
df_clean = df_clean[~df_clean[target].isna()]


# %%
X = df_clean.drop(columns=[target]).to_numpy()
y = df_clean[target].to_numpy()


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42
)

# %%
tree = ID3Tree(
    max_depth=10,
    min_samples=10
)

tree.fit(X_train, y_train)

# %%
y_pred = tree.predict(X_test)

acc_before = accuracy_score(y_test, y_pred)
mse_before = mean_squared_error(y_test, y_pred)

print("ID3 BEFORE PRUNING")
print("Accuracy:", acc_before)
print("MSE:", mse_before)

# %%
tree.prune(X_val, y_val)

y_pred_pruned = tree.predict(X_test)

acc_after = accuracy_score(y_test, y_pred_pruned)
mse_after = mean_squared_error(y_test, y_pred_pruned)

print("ID3 AFTER PRUNING")
print("Accuracy:", acc_after)
print("MSE:", mse_after)

# %%
sk_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42
)

sk_tree.fit(X_train, y_train)

y_pred_sk = sk_tree.predict(X_test)

acc_sk = accuracy_score(y_test, y_pred_sk)
mse_sk = mean_squared_error(y_test, y_pred_sk)

print("SKLEARN TREE")
print("Accuracy:", acc_sk)
print("MSE:", mse_sk)

# %%
results = pd.DataFrame({
    "Model": [
        "ID3 before pruning",
        "ID3 after pruning",
        "Sklearn DecisionTree"
    ],
    "Accuracy": [
        acc_before,
        acc_after,
        acc_sk
    ],
    "MSE": [
        mse_before,
        mse_after,
        mse_sk
    ]
})

# %%
results.sort_values("Accuracy", ascending=False)
results


