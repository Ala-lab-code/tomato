# baseline_lr.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils import compute_class_weight
from tqdm import tqdm

class ImprovedLogisticRegressionCV:
    """改进的逻辑回归模型，支持类别权重、PCA 95%方差、交叉验证"""

    def __init__(self, num_classes, class_weights=None):
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        self.best_params_ = None
        self.cv_results_ = None
        self.history = {'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
                        'train_loss': [], 'val_loss': []}

    def _prepare_data(self, data_loader):
        X_list, y_list = [], []
        for batch in tqdm(data_loader, desc="处理批次"):
            features, labels = batch
            X_list.append(features)
            y_list.append(labels)
        X = np.vstack(X_list) if len(X_list) > 1 else X_list[0]
        y = np.concatenate(y_list)
        return X, y

    def compute_optimal_class_weights(self, y_train):
        unique, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        n_classes = len(unique)
        counts = np.maximum(counts, 1)
        freq_weights = total_samples / (n_classes * counts)
        freq_weights = freq_weights / freq_weights.mean()
        optimal_weights = {class_idx: freq_weights[i] for i, class_idx in enumerate(unique)}
        return optimal_weights

    def perform_stratified_cross_validation(self, X_train, y_train, cv_folds=5):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        param_grid = {
            'C': [0.002, 0.005, 0.01, 0.02, 0.05],
            'solver': ['saga', 'lbfgs'],
            'max_iter': [1000, 2000],
            'penalty': ['l2', 'none'],
            'class_weight': [self.class_weights if self.class_weights else 'balanced']
        }
        base_model = LogisticRegression(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid,
                                   cv=skf, scoring='f1_weighted', n_jobs=-1, verbose=1, refit=True)
        grid_search.fit(X_train, y_train)
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        return grid_search.best_estimator_

    def train_with_validation(self, train_loader, val_loader, use_pca=True, cv_folds=5):
        X_train, y_train = self._prepare_data(train_loader)
        X_val, y_val = self._prepare_data(val_loader)

        if self.class_weights is None:
            self.class_weights = self.compute_optimal_class_weights(y_train)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if use_pca and X_train_scaled.shape[1] > 10:
            # 自动选择PCA组件数达到95%方差
            explained_variance_target = 0.95
            pca_full = IncrementalPCA(n_components=min(1024, X_train_scaled.shape[1]), batch_size=1024)
            X_train_pca_full = pca_full.fit_transform(X_train_scaled)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.searchsorted(cum_var, explained_variance_target) + 1
            self.pca = IncrementalPCA(n_components=n_components, batch_size=1024)
            X_train_processed = self.pca.fit_transform(X_train_scaled)
            X_val_processed = self.pca.transform(X_val_scaled)
        else:
            X_train_processed = X_train_scaled
            X_val_processed = X_val_scaled

        # 交叉验证获取最佳参数
        self.model = self.perform_stratified_cross_validation(X_train_processed, y_train, cv_folds=cv_folds)

        # 最终训练
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True

        # 训练历史
        y_train_pred = self.model.predict(X_train_processed)
        y_val_pred = self.model.predict(X_val_processed)
        self.history['train_acc'].append(accuracy_score(y_train, y_train_pred))
        self.history['val_acc'].append(accuracy_score(y_val, y_val_pred))
        self.history['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
        self.history['val_f1'].append(f1_score(y_val, y_val_pred, average='weighted'))

        return self.history

    def evaluate_on_test(self, test_loader, class_names=None):
        if not self.is_trained:
            raise ValueError("模型未训练，请先训练模型!")
        X_test, y_test = self._prepare_data(test_loader)
        X_test_scaled = self.scaler.transform(X_test)
        if self.pca is not None:
            X_test_processed = self.pca.transform(X_test_scaled)
        else:
            X_test_processed = X_test_scaled
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        if class_names:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title("混淆矩阵")
            plt.show()
        return {'accuracy': acc, 'f1_score': f1, 'precision': precision, 'recall': recall,
                'predictions': y_pred, 'probabilities': y_pred_proba, 'true_labels': y_test}

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'pca': self.pca,
                         'num_classes': self.num_classes, 'class_weights': self.class_weights}, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.num_classes = data['num_classes']
        self.class_weights = data['class_weights']
        self.is_trained = True
