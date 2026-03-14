import pandas as pd
import numpy as np
import sklearn as sk
import kagglehub
import os
from DecisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def plot_tree_matplotlib(model):
    fig, ax = plt.subplots()
    ax.axis('off')

    def draw_node(node, x, y, dx, depth=0):
        if node is None:
            return

        # Рисуем кружок/квадрат
        if node.is_leaf:
            text = f"Class: {node.predicted_class}"
            color = 'lightgreen'
        else:
            text = f"f[{node.feature}]<={node.threshold:.1f}"
            color = 'lightblue'

        ax.text(x, y, text, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=color, edgecolor='black'))

        if not node.is_leaf:
            # Линии к детям
            dy = 1.5 / (2 ** depth)
            draw_node(node.left, x - dx, y - dy, dx / 2, depth + 1)
            draw_node(node.right, x + dx, y - dy, dx / 2, depth + 1)
            ax.plot([x, x - dx], [y, y - dy], 'k-')
            ax.plot([x, x + dx], [y, y - dy], 'k-')

    draw_node(model.tree_, 0.5, 1, 0.25)
    plt.show()


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет основные метрики классификации

    Parameters:
    y_true : array-like, истинные метки классов
    y_pred : array-like, предсказанные метки классов
    """

    # Проверка на одинаковую длину массивов
    if len(y_true) != len(y_pred):
        raise ValueError("Массивы должны иметь одинаковую длину")

    # Вычисление метрик
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Вывод результатов
    print("МЕТРИКИ КЛАССИФИКАЦИИ")
    print("=" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


path = kagglehub.dataset_download("miadul/hypertension-risk-prediction-dataset")
contents = os.listdir(path)

for item in contents:
    print(item)
df = pd.read_csv(path + '/hypertension_dataset.csv')

unique_vals = df['Medication'].dropna().unique()
mapper = {val: i for i, val in enumerate(unique_vals)}
encoded = df['Medication'].map(mapper)
df['Medication'] = encoded
from sklearn.preprocessing import LabelEncoder
bp_model = LabelEncoder()
df['BP_History'] = bp_model.fit_transform(df['BP_History'])
fh_model = LabelEncoder()
df['Family_History'] = fh_model.fit_transform(df['Family_History'])
el_model = LabelEncoder()
df['Exercise_Level'] = el_model.fit_transform(df['Exercise_Level'])
ss_model = LabelEncoder()
df['Smoking_Status'] = ss_model.fit_transform(df['Smoking_Status'])
hh_model = LabelEncoder()
df['Has_Hypertension'] = hh_model.fit_transform(df['Has_Hypertension'])

y = df['Has_Hypertension']
X = df.drop('Has_Hypertension', axis='columns')
X_val = X.sample(frac=0.5, random_state=42)
y_val = y.sample(frac=0.5, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
X_np = X_train.to_numpy()
y_np = y_train.to_numpy()
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()

model1 = DecisionTree(30, 2, 1e-4, to_prune = True)
model1.fit(X_np, y_np, X_val, y_val)
y_pred1 = model1.predict(X_test)
print(calculate_metrics(y_test, y_pred1))

model2 = DecisionTree(20, 2, 1e-4, to_prune = False)
model2.fit(X_np, y_np)
y_pred2 = model2.predict(X_test)
print(calculate_metrics(y_test, y_pred2))

model3 = DecisionTreeClassifier(max_depth=20, min_samples_leaf=2)
model3.fit(X_np, y_np)
y_pred3 = model3.predict(X_test)
print(calculate_metrics(y_test, y_pred3))

plot_tree_matplotlib(model2)