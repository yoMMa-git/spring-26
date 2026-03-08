import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print("Загрузка данных...")
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

df['Loan_Status'] = (df['Loan_Status'] == 'Y').astype(int)
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1).copy()
y = df['Loan_Status'].copy()

binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Graduate': 1, 'Not Graduate': 0}
for col in ['Gender', 'Married', 'Self_Employed', 'Education']:
    X[col] = X[col].map(binary_map)
X['Dependents'] = X['Dependents'].replace({'3+': 3}).astype(float)
X['Property_Area'] = X['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)
print(f"Данные готовы: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"Пропусков в train: {X_train.isna().sum().sum()}")


class Node:
    """
    Узел решающего дерева
    """
    def __init__(self):
        self.feature_idx = None  # индекс признака fv(x) для сплита
        self.threshold = None  # порог a для бинарного признака [fj(x) >= a]
        self.left = None  # дочерняя вершина для fv(x) = 0 (< threshold)
        self.right = None  # дочерняя вершина для fv(x) = 1 (>= threshold)
        self.is_leaf = False  # флаг: v ∈ V_лист или v ∈ V_внутр
        self.label = None  # yv - предсказание для листа
        self.class_dist = None  # P(y|U) - распределение классов в узле
        self.branch_weights = None  # q_vk = |Uk|/|U| для обработки пропусков


def gini_impurity(y):
    """
    Расчёт неопределённости Джини (Gini impurity)
    Φ(U) = 4p(1-p)
    """
    if len(y) == 0:
        return 0.0
    p = np.mean(y)  # частотная оценка P(y=1|U)
    return 4 * p * (1 - p)  # Φ(U) = 4p(1-p)


def calc_gain(y_parent, y_left, y_right):
    """
    Расчёт прироста информации (Information Gain)
    Gain(f, U) = Φ(U) - Φ(U|f) → max
    """
    if len(y_parent) == 0:
        return 0.0

    parent_impurity = gini_impurity(y_parent)
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)

    if n_left == 0 or n_right == 0:
        return 0.0

    # Φ(U|f) = Σ_k (|Uk|/|U|) * Φ(Uk)
    child_impurity = (n_left / n) * gini_impurity(y_left) + (n_right / n) * gini_impurity(y_right)

    return parent_impurity - child_impurity


def find_best_split(X, y):
    """
    Поиск лучшего признака и порога для разбиения
    Реализует жадную стратегию ID3
    Обработка пропусков
    """
    best_gain = -1
    best_feature = None
    best_threshold = None

    n_features = X.shape[1]

    for feat_idx in range(n_features):
        feature_vals = X[:, feat_idx]

        # маска валидных (не пропущенных) значений
        valid_mask = ~np.isnan(feature_vals)

        if np.sum(valid_mask) < 2:
            continue

        unique_vals = np.unique(feature_vals[valid_mask])

        # если мало уникальных значений - считаем признак категориальным
        if len(unique_vals) <= 10:
            thresholds = unique_vals
        else:
            thresholds = [(unique_vals[i] + unique_vals[i + 1]) / 2 for i in range(len(unique_vals) - 1)]

        for thresh in thresholds:
            left_mask = valid_mask & (feature_vals < thresh)
            right_mask = valid_mask & (feature_vals >= thresh)

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            gain = calc_gain(y, y[left_mask], y[right_mask])

            if gain > best_gain:
                best_gain = gain
                best_feature = feat_idx
                best_threshold = thresh

    return best_feature, best_threshold, best_gain


class MyID3:
    """
    Реализация алгоритма ID3 (Iterative Dichotomiser)
    TreeGrowing(Вход: U ⊆ X^ℓ) → Выход: корень дерева v
    """
    def __init__(self, max_depth=10, min_samples=5, min_gain=0.01):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_gain = min_gain
        self.root = None
        self.feature_names = None

    def _build_tree(self, X, y, depth=0):
        """Рекурсивная функция построения дерева"""
        node = Node()

        # сохраняем распределение классов P(y|U)
        if len(y) > 0:
            p1 = np.mean(y)
            node.class_dist = np.array([1 - p1, p1])
        else:
            node.class_dist = np.array([0.5, 0.5])

        # критерии остановки
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            node.is_leaf = True
            # мажоритарное правило Major(U) := argmax_y P(y|U)
            node.label = 1 if np.mean(y) > 0.5 else 0
            return node

        # поиск лучшего сплита fv := argmax Gain(f, U)
        feat_idx, thresh, gain = find_best_split(X, y)

        if gain < self.min_gain or feat_idx is None:
            node.is_leaf = True
            node.label = 1 if np.mean(y) > 0.5 else 0
            return node

        node.feature_idx = feat_idx
        node.threshold = thresh

        # разбиение данных с учётом пропусков
        feature_vals = X[:, feat_idx]
        valid_mask = ~np.isnan(feature_vals)

        left_mask = valid_mask & (feature_vals < thresh)
        right_mask = valid_mask & (feature_vals >= thresh)

        # расчёт весов ветвей q_vk = |Uk|/|U|
        n_valid = np.sum(valid_mask)
        if n_valid > 0:
            node.branch_weights = np.array([np.sum(left_mask) / n_valid, np.sum(right_mask) / n_valid])
        else:
            node.branch_weights = np.array([0.5, 0.5])

        # рекурсивное построение дочерних вершин
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y, feature_names=None):
        """Обучение дерева v0 := TreeGrowing(X^ℓ)"""
        self.feature_names = feature_names
        self.root = self._build_tree(X, y)
        return self

    def _predict_proba_single(self, x, node):
        """
        Предсказание вероятностей для одного объекта
        """
        if node.is_leaf:
            return node.class_dist

        val = x[node.feature_idx]

        # обработка пропущенного значения
        if np.isnan(val):
            # средневзвешенное распределение по дочерним вершинам
            left_proba = self._predict_proba_single(x, node.left)
            right_proba = self._predict_proba_single(x, node.right)
            return node.branch_weights[0] * left_proba + node.branch_weights[1] * right_proba

        # переход по соответствующей ветви
        if val < node.threshold:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)

    def predict_proba(self, X):
        return np.array([self._predict_proba_single(x, self.root) for x in X])

    def predict(self, X):
        """a(x) = argmax_{y∈Y} P(y|x, v0) наиболее вероятный класс"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # реализуем pruning
    def _get_mask_for_node(self, X, node, mask=None):
        """Получить маску объектов, дошедших до узла"""
        if mask is None:
            mask = np.ones(len(X), dtype=bool)

        if node.is_leaf or node.feature_idx is None:
            return mask

        feature_vals = X[:, node.feature_idx]
        valid_mask = ~np.isnan(feature_vals) & mask

        left_mask = valid_mask & (feature_vals < node.threshold)
        right_mask = valid_mask & (feature_vals >= node.threshold)

        return left_mask | right_mask

    def _prune_recursive(self, node, X_val, y_val, val_mask):
        if node.is_leaf:
            return

        # вычисляем маски для детей
        if node.feature_idx is not None:
            feature_vals = X_val[:, node.feature_idx]
            valid_mask = ~np.isnan(feature_vals) & val_mask

            left_mask = valid_mask & (feature_vals < node.threshold)
            right_mask = valid_mask & (feature_vals >= node.threshold)
        else:
            left_mask = right_mask = val_mask

        # рекурсивно обрезаем детей
        if node.left:
            self._prune_recursive(node.left, X_val, y_val, left_mask)
        if node.right:
            self._prune_recursive(node.right, X_val, y_val, right_mask)

        # проверяем объекты в текущем узле
        y_val_node = y_val[val_mask]
        if len(y_val_node) == 0:
            return

        # ошибка текущего поддерева (только для объектов этого узла)
        y_pred_subtree = np.array([self._predict_proba_single(X_val[i], node)[1] >= 0.5 for i in range(len(X_val)) if val_mask[i]])
        err_subtree = 1 - accuracy_score(y_val_node, y_pred_subtree)

        saved_left = node.left
        saved_right = node.right
        saved_is_leaf = node.is_leaf
        saved_label = node.label
        saved_dist = node.class_dist

        # заменяем на лист
        node.left = None
        node.right = None
        node.is_leaf = True

        # мажоритарный класс из training distribution
        p1 = node.class_dist[1]
        node.label = 1 if p1 >= 0.5 else 0

        # ошибка после pruning
        y_pred_leaf = np.full(len(y_val_node), node.label)
        err_leaf = 1 - accuracy_score(y_val_node, y_pred_leaf)

        # если стало хуже - откатываем
        if err_leaf > err_subtree:
            node.left = saved_left
            node.right = saved_right
            node.is_leaf = saved_is_leaf
            node.label = saved_label
            node.class_dist = saved_dist

    def prune(self, X_val, y_val):
        """Запуск pruning"""
        val_mask = np.ones(len(X_val), dtype=bool)
        self._prune_recursive(self.root, X_val, y_val, val_mask)


def extract_rules(node, feature_names, path=None, rules=None):
    """Извлечение логических правил из дерева"""
    if path is None:
        path = []
    if rules is None:
        rules = []

    if node.is_leaf:
        rule = {
            'conditions': path.copy(),
            'label': node.label,
            'prob': node.class_dist[1],
            'depth': len(path)
        }
        rules.append(rule)
        return rules

    feat_name = feature_names[node.feature_idx] if feature_names else f"f{node.feature_idx}"
    thresh = node.threshold
    extract_rules(node.left, feature_names, path + [f"{feat_name} < {thresh:.1f}"], rules)
    extract_rules(node.right, feature_names, path + [f"{feat_name} >= {thresh:.1f}"], rules)
    return rules


def print_rules(rules, top_k=5):
    print(f"\nТоп-{top_k} извлечённых правил:")
    print("-" * 60)

    rules_sorted = sorted(rules, key=lambda r: abs(r['prob'] - 0.5) * (1 / (r['depth'] + 1)), reverse=True)

    for i, rule in enumerate(rules_sorted[:top_k], 1):
        conditions = " И ".join(rule['conditions']) if rule['conditions'] else "Без условий"
        if rule['prob'] >= 0.5:
            label_str = "ОДОБРЕН"
            true_label = 1
        else:
            label_str = "ОТКАЗ"
            true_label = 0

        print(f"{i}. ЕСЛИ [{conditions}]")
        print(f"    -> {label_str} (класс={true_label}, вероятность={rule['prob']:.2f}, глубина={rule['depth']})")
        print()


def visualize_tree(node, feature_names, depth=0, pos=0.5, width=0.3, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Структура решающего дерева', fontsize=14, fontweight='bold', pad=20)

    if node.is_leaf:
        predicted_class = 1 if node.class_dist[1] >= 0.5 else 0
        label_text = "ОДОБРЕН" if predicted_class == 1 else "ОТКАЗ"
        prob_text = f"p={node.class_dist[1]:.2f}"
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightgreen",
                          edgecolor="green", linewidth=2)
        ax.text(pos, 0.9 - depth * 0.15, f"{label_text}\n({prob_text})",
                bbox=bbox_props, ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        feat_name = feature_names[node.feature_idx] if feature_names else f"f{node.feature_idx}"
        threshold = node.threshold

        if feat_name == 'Credit_History':
            feat_short = 'Credit_Hist'
        elif feat_name == 'ApplicantIncome':
            feat_short = 'Income'
        elif feat_name == 'LoanAmount':
            feat_short = 'Loan_Amt'
        else:
            feat_short = feat_name

        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightblue",
                          edgecolor="blue", linewidth=2)
        ax.text(pos, 0.9 - depth * 0.15, f"{feat_short} < {threshold:.1f}",
                bbox=bbox_props, ha='center', va='center', fontsize=9)

        left_pos = pos - width/2
        right_pos = pos + width/2

        ax.plot([pos, left_pos], [0.9 - depth * 0.15, 0.9 - (depth + 1) * 0.15], 'k-', alpha=0.6)
        ax.plot([pos, right_pos], [0.9 - depth * 0.15, 0.9 - (depth + 1) * 0.15], 'k-', alpha=0.6)

        ax.text((pos + left_pos) / 2 - 0.03, 0.9 - (depth + 0.5) * 0.15, 'Yes',
                fontsize=8, fontstyle='italic', alpha=0.7)
        ax.text((pos + right_pos) / 2 + 0.01, 0.9 - (depth + 0.5) * 0.15, 'No',
                fontsize=8, fontstyle='italic', alpha=0.7)

        if node.left:
            visualize_tree(node.left, feature_names, depth + 1, left_pos, width * 0.6, ax)
        if node.right:
            visualize_tree(node.right, feature_names, depth + 1, right_pos, width * 0.6, ax)

    return ax


print("\n Обучение собственного дерева ID3...")

feature_names = X.columns.tolist()
X_train_arr = X_train.values.astype(float)
X_val_arr = X_val.values.astype(float)
X_test_arr = X_test.values.astype(float)

# обучаем дерево
model = MyID3(max_depth=8, min_samples=10, min_gain=0.02)
model.fit(X_train_arr, y_train.values, feature_names=feature_names)

print("\nМетрики ДО pruning:")
y_pred_train = model.predict(X_train_arr)
y_pred_val = model.predict(X_val_arr)
y_pred_test = model.predict(X_test_arr)

print(f"    Train: Acc={accuracy_score(y_train, y_pred_train):.3f}")
print(f"    Val: Acc={accuracy_score(y_val, y_pred_val):.3f}")
print(f"    Test: Acc={accuracy_score(y_test, y_pred_test):.3f}, F1={f1_score(y_test, y_pred_test):.3f}")

visualize_tree(model.root, feature_names)
plt.show()

print("\nПрименение Reduced Error Pruning...")
model.prune(X_val_arr, y_val.values)

print("\nМетрики ПОСЛЕ pruning:")
y_pred_test_pruned = model.predict(X_test_arr)
print(f"    Test: Acc={accuracy_score(y_test, y_pred_test_pruned):.3f}, F1={f1_score(y_test, y_pred_test_pruned):.3f}")

print("\nСравнение с sklearn.DecisionTreeClassifier...")
X_train_filled = X_train.copy()
X_val_filled = X_val.copy()
X_test_filled = X_test.copy()

for col in X_train_filled.columns:
    if X_train_filled[col].isna().any():
        if X_train_filled[col].dtype in [np.float64, np.int64]:
            fill_val = X_train_filled[col].median()
        else:
            fill_val = X_train_filled[col].mode()[0]

        X_train_filled[col] = X_train_filled[col].fillna(fill_val)
        X_val_filled[col] = X_val_filled[col].fillna(fill_val)
        X_test_filled[col] = X_test_filled[col].fillna(fill_val)

sklearn_model = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=10, random_state=42)
sklearn_model.fit(X_train_filled, y_train)

y_pred_sklearn = sklearn_model.predict(X_test_filled)
print(f"    Sklearn Test: Acc={accuracy_score(y_test, y_pred_sklearn):.3f}, F1={f1_score(y_test, y_pred_sklearn):.3f}")

rules = extract_rules(model.root, feature_names)
print_rules(rules, top_k=5)

print("\nРезультаты сравнения:")
print(f"{'Модель':<25} {'Test Acc':<12} {'Test F1':<12}")
print("-" * 50)
print(f"{'Custom ID3 (до pruning)':<25} {accuracy_score(y_test, y_pred_test):.3f}{'':<7} {f1_score(y_test, y_pred_test):.3f}")
print(f"{'Custom ID3 (после pruning)':<25} {accuracy_score(y_test, y_pred_test_pruned):.3f}{'':<7} {f1_score(y_test, y_pred_test_pruned):.3f}")
print(f"{'Sklearn (эталон)':<25} {accuracy_score(y_test, y_pred_sklearn):.3f}{'':<7} {f1_score(y_test, y_pred_sklearn):.3f}")


def plot_confusion_matrix(y_test, y_pred_before, y_pred_after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_before = confusion_matrix(y_test, y_pred_before)
    cm_after = confusion_matrix(y_test, y_pred_after)

    im1 = axes[0].imshow(cm_before, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('До pruning', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Предсказанный класс', fontsize=10)
    axes[0].set_ylabel('Истинный класс', fontsize=10)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(cm_after, interpolation='nearest', cmap=plt.cm.Greens)
    axes[1].set_title('После pruning', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Предсказанный класс', fontsize=10)
    axes[1].set_ylabel('Истинный класс', fontsize=10)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    for i, cm in enumerate([cm_before, cm_after]):
        ax = axes[i]
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, str(cm[j, k]), ha='center', va='center', color='white' if cm[j, k] > cm.max() / 2 else 'black',
                        fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


plot_confusion_matrix(y_test.values, y_pred_test, y_pred_test_pruned)


def plot_metrics_vs_depth(X_train, y_train, X_test, y_test, max_depth=10):
    train_accs = []
    test_accs = []
    depths = list(range(1, max_depth + 1))

    print("\nОбучение деревьев разной глубины...")
    for depth in depths:
        model_temp = MyID3(max_depth=depth, min_samples=5, min_gain=0.01)
        model_temp.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, model_temp.predict(X_train)))
        test_accs.append(accuracy_score(y_test, model_temp.predict(X_test)))

    optimal_depth = depths[np.argmax(test_accs)]
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accs, marker='o', linewidth=2, label="Train Accuracy")
    plt.plot(depths, test_accs, marker='s', linewidth=2, label="Test Accuracy")
    plt.axvline(optimal_depth, linestyle="--", linewidth=2, label=f"Best depth = {optimal_depth}")
    plt.xlabel("Глубина дерева")
    plt.ylabel("Accuracy")
    plt.title("Переобучение в зависимости от глубины дерева")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_metrics_vs_depth(X_train_arr, y_train.values, X_test_arr, y_test.values, max_depth=8)

visualize_tree(model.root, feature_names)
plt.show()


def plot_feature_importance(node, feature_names, depth=0, importance_dict=None):
    if importance_dict is None:
        importance_dict = {}

    if node.is_leaf:
        return importance_dict

    feat_idx = node.feature_idx
    feat_name = feature_names[feat_idx] if feature_names else f"f{feat_idx}"

    weight = 1.0 / (depth + 1)
    importance_dict[feat_name] = importance_dict.get(feat_name, 0) + weight

    if node.left:
        plot_feature_importance(node.left, feature_names, depth + 1, importance_dict)
    if node.right:
        plot_feature_importance(node.right, feature_names, depth + 1, importance_dict)

    return importance_dict


importance = plot_feature_importance(model.root, feature_names)

if importance:
    plt.figure(figsize=(10, 6))
    features = list(importance.keys())
    values = list(importance.values())
    values = [v / sum(values) for v in values]

    bars = plt.barh(features, values, color='skyblue', edgecolor='blue', alpha=0.8)
    plt.xlabel('Важность', fontsize=11)
    plt.title('Важность признаков в решающем дереве', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.show()

    print(f"\nВажность признаков:")
    for feat, val in sorted(zip(features, values), key=lambda x: x[1], reverse=True):
        print(f"   {feat}: {val:.2%}")
