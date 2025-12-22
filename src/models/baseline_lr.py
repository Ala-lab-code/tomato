# lr_baseline.py
import os
import time
import pickle
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.utils import compute_class_weight


class ImprovedLogisticRegressionCV:
    """改进的逻辑回归模型，支持类别权重、交叉验证和超参数调优"""

    def __init__(self, num_classes: int, class_weights=None):
        """
        初始化改进的逻辑回归模型

        Args:
            num_classes: 类别数量
            class_weights: 类别权重，可以是字典或数组
        """
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        self.best_params_ = None
        self.cv_results_ = None
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'train_loss': [],
            'val_loss': []
        }

        print(f"改进的逻辑回归模型初始化:")
        print(f"  类别数: {num_classes}")
        print(f"  类别权重: {'已设置' if class_weights is not None else '未设置'}")
        print(f"  支持: 5-Fold交叉验证, 超参数调优, 学习曲线可视化")

    def _prepare_data(self, data_loader, with_weights=False):
        """从数据加载器中提取特征、标签和权重"""
        X_list = []
        y_list = []
        w_list = []

        print("提取数据特征...")
        for batch in tqdm(data_loader, desc="处理批次"):
            if with_weights and len(batch) == 3:
                features, labels, weights = batch
                X_list.append(features)
                y_list.append(labels)
                w_list.append(weights)
            else:
                features, labels = batch
                X_list.append(features)
                y_list.append(labels)

        X = np.vstack(X_list) if len(X_list) > 1 else X_list[0]
        y = np.concatenate(y_list)

        if with_weights and len(w_list) > 0:
            weights = np.concatenate(w_list)
            return X, y, weights
        else:
            return X, y, None

    def compute_optimal_class_weights(self, y_train):
        """计算最优类别权重"""
        print("计算最优类别权重...")

        # 方法1: sklearn的compute_class_weight
        sklearn_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(self.num_classes),
            y=y_train
        )

        # 方法2: 基于样本数的权重
        unique, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        n_classes = len(unique)

        # 避免除以零
        counts = np.maximum(counts, 1)

        # 计算权重: 总样本数 / (类别数 * 类别样本数)
        freq_weights = total_samples / (n_classes * counts)

        # 归一化权重
        freq_weights = freq_weights / freq_weights.mean()

        # 创建权重数组
        optimal_weights = {}
        for i, class_idx in enumerate(unique):
            optimal_weights[class_idx] = freq_weights[i]

        print(f"类别权重计算完成:")
        for class_idx in range(self.num_classes):
            weight = optimal_weights.get(class_idx, 1.0)
            count = np.sum(y_train == class_idx) if class_idx in unique else 0
            print(f"  类别 {class_idx}: {count}个样本, 权重={weight:.3f}")

        return optimal_weights

    def perform_stratified_cross_validation(self, X_train, y_train, cv_folds=5):
        """
        执行分层交叉验证和超参数调优

        Args:
            X_train: 训练特征
            y_train: 训练标签
            cv_folds: 交叉验证折数

        Returns:
            最佳模型和结果
        """
        print(f"\n执行分层 {cv_folds}-Fold 交叉验证...")

        # 创建分层KFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # 定义超参数网格
        if self.class_weights is not None:
            param_grid = {
                'C': [0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05],
                'solver': ['saga', 'lbfgs', 'newton-cg'],
                'max_iter': [1000, 2000, 5000],
                'penalty': ['l2', 'none'],
                'class_weight': [self.class_weights, 'balanced']
            }
        else:
            param_grid = {
                'C': [0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05],
                'solver': ['saga', 'lbfgs', 'newton-cg'],
                'max_iter': [1000, 2000, 5000],
                'penalty': ['l2', 'none'],
                'class_weight': ['balanced', None]
            }

        # 创建基础模型
        base_model = LogisticRegression(random_state=42, n_jobs=-1)

        print("使用网格搜索进行超参数调优...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=skf,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        # 执行搜索
        grid_search.fit(X_train, y_train)

        print(f"\n交叉验证结果:")
        print(f"  最佳参数: {grid_search.best_params_}")
        print(f"  最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

        # 保存结果
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_

        return grid_search.best_estimator_, grid_search.best_score_

    def train_with_validation(self, train_loader, val_loader,
                              train_weights=None,
                              use_pca=True, pca_components_range=[50, 100, 150, 200],
                              cv_folds=5, early_stopping=True, patience=10):
        """
        训练模型，包含类别权重、交叉验证和超参数调优

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            train_weights: 训练样本权重
            use_pca: 是否使用PCA
            pca_components_range: PCA组件数搜索范围
            cv_folds: 交叉验证折数
            early_stopping: 是否使用早停
            patience: 早停耐心值

        Returns:
            训练结果字典
        """
        print("=" * 80)
        print("开始训练改进的逻辑回归模型（包含类别权重和交叉验证）")
        print("=" * 80)

        start_time = time.time()

        # 1. 提取训练和验证数据
        print("\n1. 提取数据...")
        X_train, y_train, _ = self._prepare_data(train_loader)
        X_val, y_val, _ = self._prepare_data(val_loader)

        print(f"训练数据: {X_train.shape[0]:,} 个样本, {X_train.shape[1]} 个特征")
        print(f"验证数据: {X_val.shape[0]:,} 个样本, {X_val.shape[1]} 个特征")

        # 显示类别分布
        train_class_counts = np.bincount(y_train)
        print(f"\n训练集类别分布:")
        for i, count in enumerate(train_class_counts):
            print(f"  类别 {i}: {count}个样本 ({count / len(y_train) * 100:.1f}%)")

        # 2. 如果没有提供类别权重，自动计算
        if self.class_weights is None:
            self.class_weights = self.compute_optimal_class_weights(y_train)

        # 3. 从训练集计算标准化参数（防止数据泄露）
        print("\n3. 从训练集计算标准化参数...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 4. PCA降维调优（可选）
        if use_pca and X_train_scaled.shape[1] > 50:
            print(f"\n4. PCA降维调优...")
            print(f"  搜索PCA组件数: {pca_components_range}")

            best_pca_score = 0
            best_pca_model = None
            best_pca_n_components = None

            pca_results = []

            for n_components in tqdm(pca_components_range, desc="PCA调优"):
                n_components = min(n_components, X_train_scaled.shape[1])
                if n_components < 10:
                    continue

                # 创建PCA模型
                pca = IncrementalPCA(n_components=n_components, batch_size=1024)
                X_train_pca = pca.fit_transform(X_train_scaled)
                explained_variance = pca.explained_variance_ratio_.sum()

                # 执行交叉验证
                best_model, best_score = self.perform_stratified_cross_validation(
                    X_train_pca, y_train, cv_folds=cv_folds
                )

                pca_results.append({
                    'n_components': n_components,
                    'score': best_score,
                    'explained_variance': explained_variance
                })

                if best_score > best_pca_score:
                    best_pca_score = best_score
                    best_pca_model = pca
                    best_pca_n_components = n_components

            # 选择最佳PCA
            if best_pca_model is not None:
                self.pca = best_pca_model
                X_train_processed = self.pca.transform(X_train_scaled)
                X_val_processed = self.pca.transform(X_val_scaled)
                print(
                    f"\n  选择PCA组件数: {best_pca_n_components}, 解释方差: {self.pca.explained_variance_ratio_.sum():.4f}")
            else:
                X_train_processed = X_train_scaled
                X_val_processed = X_val_scaled

            # 保存PCA结果
            self.pca_results = pca_results

        else:
            X_train_processed = X_train_scaled
            X_val_processed = X_val_scaled

        # 5. 最终模型训练（使用交叉验证的最佳参数）
        print("\n5. 训练最终模型...")

        # 如果已经有最佳参数，使用它们
        if self.best_params_ is not None:
            print(f"使用最佳参数训练模型: {self.best_params_}")

            # 确保类别权重正确设置
            if 'class_weight' in self.best_params_ and self.best_params_['class_weight'] == 'from_dataset':
                self.best_params_['class_weight'] = self.class_weights

            self.model = LogisticRegression(**self.best_params_, random_state=42, n_jobs=-1)
        else:
            # 执行交叉验证获取最佳参数
            print("执行最终交叉验证获取最佳参数...")
            best_model, best_score = self.perform_stratified_cross_validation(
                X_train_processed, y_train, cv_folds=cv_folds
            )
            self.model = best_model

        # 6. 记录训练过程
        print("\n6. 记录训练过程...")
        train_start = time.time()

        if early_stopping and self.model.max_iter > 1:
            print("  使用早停机制...")

            # 使用warm_start记录训练过程
            temp_model_params = self.model.get_params()
            temp_model_params['max_iter'] = 1
            temp_model_params['warm_start'] = True

            temp_model = LogisticRegression(**temp_model_params)

            best_val_f1 = 0
            best_model_state = None
            no_improve_count = 0

            for epoch in range(self.model.max_iter):
                # 训练一个迭代
                temp_model.fit(X_train_processed, y_train)

                # 计算训练指标
                y_train_pred = temp_model.predict(X_train_processed)
                train_acc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')

                # 计算训练损失（对数损失）
                y_train_proba = temp_model.predict_proba(X_train_processed)
                train_loss = -np.mean(np.log(y_train_proba[np.arange(len(y_train)), y_train] + 1e-10))

                # 计算验证指标
                y_val_pred = temp_model.predict(X_val_processed)
                val_acc = accuracy_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred, average='weighted')

                # 计算验证损失
                y_val_proba = temp_model.predict_proba(X_val_processed)
                val_loss = -np.mean(np.log(y_val_proba[np.arange(len(y_val)), y_val] + 1e-10))

                # 记录历史
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                self.history['train_f1'].append(train_f1)
                self.history['val_f1'].append(val_f1)
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                # 检查早停（基于F1分数）
                if val_f1 > best_val_f1 + 0.0001:
                    best_val_f1 = val_f1
                    best_model_state = pickle.loads(pickle.dumps(temp_model))
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}: 训练准确率={train_acc:.4f}, 训练F1={train_f1:.4f}, "
                          f"验证准确率={val_acc:.4f}, 验证F1={val_f1:.4f}")

                if no_improve_count >= patience:
                    print(f"  早停在epoch {epoch + 1}, 验证F1: {best_val_f1:.4f}")
                    self.model = best_model_state
                    break

            if best_model_state is None:
                self.model = temp_model

        else:
            # 不使用早停，直接训练
            self.model.fit(X_train_processed, y_train)

            # 计算最终指标
            y_train_pred = self.model.predict(X_train_processed)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')

            y_train_proba = self.model.predict_proba(X_train_processed)
            train_loss = -np.mean(np.log(y_train_proba[np.arange(len(y_train)), y_train] + 1e-10))

            y_val_pred = self.model.predict(X_val_processed)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            y_val_proba = self.model.predict_proba(X_val_processed)
            val_loss = -np.mean(np.log(y_val_proba[np.arange(len(y_val)), y_val] + 1e-10))

            # 记录历史
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

        training_time = time.time() - train_start
        self.is_trained = True

        # 7. 绘制学习曲线
        print("\n7. 绘制学习曲线...")
        self._plot_learning_curves()

        # 8. 最终评估
        print("\n8. 最终模型评估...")
        y_train_pred = self.model.predict(X_train_processed)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')

        y_val_pred = self.model.predict(X_val_processed)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_precision = precision_score(y_val, y_val_pred, average='weighted')
        val_recall = recall_score(y_val, y_val_pred, average='weighted')

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        print(f"总时间: {total_time:.1f}秒 (训练: {training_time:.1f}秒)")
        print(f"\n训练集指标:")
        print(f"  准确率: {train_accuracy:.4f}")
        print(f"  F1分数: {train_f1:.4f}")
        print(f"  精确率: {train_precision:.4f}")
        print(f"  召回率: {train_recall:.4f}")

        print(f"\n验证集指标:")
        print(f"  准确率: {val_accuracy:.4f}")
        print(f"  F1分数: {val_f1:.4f}")
        print(f"  精确率: {val_precision:.4f}")
        print(f"  召回率: {val_recall:.4f}")

        print(f"\n最佳参数: {self.model.get_params()}")

        # 分析过拟合/欠拟合
        self._analyze_fit(train_accuracy, val_accuracy, train_f1, val_f1)

        return {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'training_time': total_time,
            'model_trained': True,
            'best_params': self.best_params_,
            'cv_results': self.cv_results_,
            'history': self.history,
            'feature_dim': X_train.shape[1],
            'class_weights': self.class_weights
        }

    def _plot_learning_curves(self):
        """绘制学习曲线"""
        if len(self.history['train_acc']) == 0:
            print("没有训练历史可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        epochs = range(1, len(self.history['train_acc']) + 1)

        # 准确率曲线
        axes[0, 0].plot(epochs, self.history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_title('训练和验证准确率曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1分数曲线
        axes[0, 1].plot(epochs, self.history['train_f1'], 'b-', label='训练F1分数', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_f1'], 'r-', label='验证F1分数', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1分数')
        axes[0, 1].set_title('训练和验证F1分数曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 损失曲线
        if len(self.history['train_loss']) > 0 and max(self.history['train_loss']) > 0:
            axes[1, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
            axes[1, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('损失')
            axes[1, 0].set_title('训练和验证损失曲线')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 类别权重可视化
        if self.class_weights is not None:
            axes[1, 1].bar(range(len(self.class_weights)), list(self.class_weights.values()))
            axes[1, 1].set_xlabel('类别索引')
            axes[1, 1].set_ylabel('权重')
            axes[1, 1].set_title('类别权重分布')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _analyze_fit(self, train_acc, val_acc, train_f1, val_f1):
        """分析过拟合/欠拟合"""
        acc_gap = train_acc - val_acc
        f1_gap = train_f1 - val_f1

        print("\n过拟合/欠拟合分析:")
        print(f"  训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}, 差距: {acc_gap:.4f}")
        print(f"  训练F1分数: {train_f1:.4f}, 验证F1分数: {val_f1:.4f}, 差距: {f1_gap:.4f}")

        if acc_gap > 0.15:
            print("  ⚠️  严重过拟合: 训练-验证准确率差距大于15%")
            print("  建议: 增加正则化(减小C), 添加更多数据, 使用更简单的模型")
        elif acc_gap > 0.08:
            print("  ⚠️  可能存在过拟合: 训练-验证准确率差距大于8%")
            print("  建议: 增加正则化, 使用早停, 数据增强")
        elif acc_gap < 0.02:
            print("  ✅  拟合良好: 训练-验证准确率差距小于2%")
        elif val_acc < 0.6 and train_acc < 0.6:
            print("  ⚠️  可能存在欠拟合: 训练和验证准确率都较低")
            print("  建议: 增加模型复杂度, 减少正则化(增大C), 增加训练时间")
        else:
            print("  ℹ️  模型表现正常")

        # 检查类别权重是否有效
        if self.class_weights is not None:
            weights = list(self.class_weights.values())
            if max(weights) > 5 * min(weights):
                print("  ⚠️  类别权重差异较大，可能存在严重类别不平衡")

    def plot_cv_results(self):
        """绘制交叉验证结果"""
        if self.cv_results_ is None:
            print("没有交叉验证结果可绘制")
            return

        # 绘制超参数C的影响
        if 'param_C' in self.cv_results_:
            cv_scores = self.cv_results_['mean_test_score']
            param_C = self.cv_results_['param_C']

            # 转换C值为数值
            C_values = []
            for c in param_C:
                try:
                    C_values.append(float(c))
                except:
                    C_values.append(1.0)

            plt.figure(figsize=(10, 6))
            plt.semilogx(C_values, cv_scores, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('正则化参数C (log scale)')
            plt.ylabel('交叉验证F1分数')
            plt.title('正则化参数C对模型性能的影响')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def evaluate_on_test(self, test_loader, class_names=None, idx_to_class=None):
        """在测试集上评估模型（只能使用一次）"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先训练模型!")

        print("\n" + "=" * 80)
        print("⚠️ 在测试集上最终评估（测试集只能使用一次）")
        print("=" * 80)

        start_time = time.time()

        # 1. 提取测试数据
        X_test, y_test, _ = self._prepare_data(test_loader)
        print(f"测试数据: {X_test.shape[0]:,} 个样本")

        # 2. 使用训练集的标准化和PCA参数（防止数据泄露）
        X_test_scaled = self.scaler.transform(X_test)
        if self.pca is not None:
            X_test_processed = self.pca.transform(X_test_scaled)
            print(f"使用PCA降维: {X_test_scaled.shape[1]} -> {self.pca.n_components_} 维度")
        else:
            X_test_processed = X_test_scaled

        # 3. 预测
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)

        # 4. 计算指标
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')

        eval_time = time.time() - start_time

        print(f"\n测试集指标:")
        print(f"  准确率: {test_accuracy:.4f}")
        print(f"  F1分数: {test_f1:.4f}")
        print(f"  精确率: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  评估时间: {eval_time:.1f}秒")

        # 分类报告
        if class_names is not None:
            # 如果提供了idx_to_class，使用它来映射
            if idx_to_class is not None:
                target_names = []
                for i in range(len(class_names)):
                    target_names.append(idx_to_class.get(i, f"Class_{i}"))
            else:
                target_names = class_names

            report = classification_report(y_test, y_pred, target_names=target_names,
                                           digits=4, zero_division=0)
        else:
            report = classification_report(y_test, y_pred, digits=4, zero_division=0)

        print("\n详细分类报告:")
        print(report)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        if class_names is not None:
            # 使用映射后的类别名称
            if idx_to_class is not None:
                display_labels = []
                for i in range(len(class_names)):
                    display_labels.append(idx_to_class.get(i, f"Class_{i}"))
            else:
                display_labels = class_names

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=display_labels, yticklabels=display_labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('测试集混淆矩阵')
        plt.tight_layout()
        plt.show()

        # 每类准确率和F1分数
        class_acc = []
        class_f1 = []

        for i in range(self.num_classes):
            if i < cm.shape[0] and i < cm.shape[1]:
                if cm.sum(axis=1)[i] > 0:
                    acc = cm[i, i] / cm.sum(axis=1)[i]
                else:
                    acc = 0.0
                class_acc.append(acc)

                # 计算每类的F1分数
                y_true_i = (y_test == i)
                y_pred_i = (y_pred == i)
                if np.sum(y_true_i) > 0 or np.sum(y_pred_i) > 0:
                    f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
                else:
                    f1 = 0.0
                class_f1.append(f1)

        print("\n每类性能:")
        for i, (acc, f1) in enumerate(zip(class_acc, class_f1)):
            if idx_to_class is not None and i in idx_to_class:
                class_name = idx_to_class[i]
            elif class_names is not None and i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f"Class_{i}"

            samples = np.sum(y_test == i)
            weight = self.class_weights.get(i, 1.0) if self.class_weights else 1.0
            print(f"  {class_name}: {samples}个样本, 权重={weight:.3f}, 准确率={acc:.4f}, F1分数={f1:.4f}")

        return {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'confusion_matrix': cm,
            'class_accuracy': class_acc,
            'class_f1': class_f1,
            'classification_report': report,
            'evaluation_time': eval_time
        }

    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'num_classes': self.num_classes,
            'class_weights': self.class_weights,
            'is_trained': self.is_trained,
            'best_params': self.best_params_,
            'history': self.history,
            'cv_results': self.cv_results_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.num_classes = model_data['num_classes']
        self.class_weights = model_data.get('class_weights', None)
        self.is_trained = model_data['is_trained']
        self.best_params_ = model_data.get('best_params', None)
        self.history = model_data.get('history',
                                      {'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
                                       'train_loss': [], 'val_loss': []})
        self.cv_results_ = model_data.get('cv_results', None)

        print(f"模型已从 {filepath} 加载")


def main():
    """主函数"""
    # 导入features模块中的函数
    from src.features import create_dataloaders

    # 数据路径
    PROCESSED_DATA_DIR = r"C:\Users\someb\Desktop\tomato_disease_classification\data\processed"

    # 加载划分元数据
    metadata_file = os.path.join(PROCESSED_DATA_DIR, "split_metadata.json")
    split_metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            split_metadata = json.load(f)

    # 加载类别权重
    weights_file = os.path.join(PROCESSED_DATA_DIR, "class_weights.npy")
    class_weights = {}
    if os.path.exists(weights_file):
        class_weights = np.load(weights_file, allow_pickle=True).item()

    # 加载类别顺序
    class_order_file = os.path.join(PROCESSED_DATA_DIR, "class_order.txt")

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, loaded_class_names, feature_dim, class_to_idx, sample_weights_info = create_dataloaders(
        data_dir=PROCESSED_DATA_DIR,
        img_size=128,
        class_weights=class_weights,
        split_metadata=split_metadata,
        class_order_file=class_order_file,
        batch_size=32
    )

    # 初始化改进的逻辑回归模型
    num_classes = len(loaded_class_names)

    # 准备类别权重
    if class_weights:
        class_weights_idx = {}
        for class_name, weight in class_weights.items():
            if class_name in class_to_idx:
                class_weights_idx[class_to_idx[class_name]] = weight
    else:
        class_weights_idx = None

    improved_model = ImprovedLogisticRegressionCV(
        num_classes=num_classes,
        class_weights=class_weights_idx
    )

    # 训练模型
    print("\n开始训练模型...")
    print("-" * 60)

    training_results = improved_model.train_with_validation(
        train_loader=train_loader,
        val_loader=val_loader,
        use_pca=True,
        pca_components_range=[100, 200, 300, 400],
        cv_folds=5,
        early_stopping=True,
        patience=10
    )

    # 绘制交叉验证结果
    print("\n绘制交叉验证结果...")
    print("-" * 60)
    improved_model.plot_cv_results()

    # 在测试集上评估模型
    print("\n在测试集上进行最终评估...")
    print("-" * 60)
    print("⚠️ 注意：测试集只能使用一次进行最终评估！")
    print("-" * 60)

    test_results = improved_model.evaluate_on_test(
        test_loader=test_loader,
        class_names=loaded_class_names
    )

    # 保存模型
    print("\n保存训练好的模型...")
    print("-" * 60)

    model_save_path = "improved_logistic_regression_cv_model.pkl"
    improved_model.save_model(model_save_path)

    # 结果分析
    print("\n模型训练和评估完成!")
    print("=" * 80)

    print(f"训练准确率: {training_results['train_accuracy']:.4f}")
    print(f"验证准确率: {training_results['val_accuracy']:.4f}")
    print(f"测试准确率: {test_results['accuracy']:.4f}")

    # 分析结果
    train_val_gap = training_results['train_accuracy'] - training_results['val_accuracy']
    test_val_gap = test_results['accuracy'] - training_results['val_accuracy']

    print(f"\n结果分析:")
    print(f"  训练-验证差距: {train_val_gap:.4f}")
    print(f"  验证-测试差距: {test_val_gap:.4f}")

    if abs(test_val_gap) > 0.05:
        print(f"  ⚠️  注意：验证集和测试集性能差距较大 ({test_val_gap:.4f})")
        print("  可能原因：验证集和测试集分布不同，或验证集过小")
    else:
        print(f"  ✅  验证集和测试集性能一致")

    # 显示训练时间
    print(f"\n训练时间: {training_results['training_time']:.1f}秒")
    print(f"测试时间: {test_results['evaluation_time']:.1f}秒")

    return improved_model, training_results, test_results


if __name__ == "__main__":
    improved_model, training_results, test_results = main()