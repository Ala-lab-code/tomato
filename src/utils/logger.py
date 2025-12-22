import logging
import os
import sys
from datetime import datetime
import json
import yaml


class ProjectLogger:
    """项目日志记录器"""

    def __init__(self, log_dir, project_name="TomatoDisease", level=logging.INFO):
        self.log_dir = log_dir
        self.project_name = project_name

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{project_name}_{timestamp}.log")

        # 配置日志
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(level)

        # 清除已有处理器
        self.logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logger initialized. Log file: {log_file}")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def log_config(self, config):
        """记录配置信息"""
        self.info("=" * 60)
        self.info("CONFIGURATION")
        self.info("=" * 60)

        # 记录配置字典
        config_str = yaml.dump(config, default_flow_style=False)
        self.info(f"\n{config_str}")

        # 保存配置到文件
        config_file = os.path.join(self.log_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.info(f"Configuration saved to: {config_file}")

    def log_metrics(self, metrics, stage="training"):
        """记录指标"""
        self.info(f"\n{'=' * 60}")
        self.info(f"{stage.upper()} METRICS")
        self.info(f"{'=' * 60}")

        # 保存指标到文件
        metrics_file = os.path.join(self.log_dir, f"{stage}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # 记录到日志
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.info(f"{key}: {value:.4f}")
                elif isinstance(value, dict):
                    self.info(f"\n{key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            self.info(f"  {sub_key}: {sub_value:.4f}")
                        else:
                            self.info(f"  {sub_key}: {sub_value}")
                else:
                    self.info(f"{key}: {value}")

        self.info(f"Metrics saved to: {metrics_file}")

    def log_experiment(self, experiment_info):
        """记录实验信息"""
        self.info(f"\n{'=' * 60}")
        self.info("EXPERIMENT INFO")
        self.info(f"{'=' * 60}")

        for key, value in experiment_info.items():
            self.info(f"{key}: {value}")

        # 保存实验信息
        info_file = os.path.join(self.log_dir, "experiment_info.json")
        with open(info_file, 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)


class TensorBoardLogger:
    """TensorBoard日志记录器"""

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def log_scalar(self, tag, value, step=None):
        """记录标量"""
        if step is None:
            step = self.step
            self.step += 1

        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step=None):
        """记录多个标量"""
        if step is None:
            step = self.step
            self.step += 1

        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag, image, step=None, dataformats='CHW'):
        """记录图像"""
        if step is None:
            step = self.step
            self.step += 1

        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images(self, tag, images, step=None, dataformats='NCHW'):
        """记录多个图像"""
        if step is None:
            step = self.step
            self.step += 1

        self.writer.add_images(tag, images, step, dataformats=dataformats)

    def log_histogram(self, tag, values, step=None):
        """记录直方图"""
        if step is None:
            step = self.step
            self.step += 1

        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_tensor):
        """记录模型图"""
        self.writer.add_graph(model, input_tensor)

    def log_embedding(self, features, metadata=None, label_img=None, tag='embedding'):
        """记录嵌入"""
        self.writer.add_embedding(features, metadata=metadata,
                                  label_img=label_img, tag=tag)

    def log_hparams(self, hparam_dict, metric_dict):
        """记录超参数"""
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        """关闭writer"""
        self.writer.close()


class ExperimentTracker:
    """实验跟踪器"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 初始化日志记录器
        self.text_logger = ProjectLogger(log_dir)
        self.tb_logger = TensorBoardLogger(log_dir)

        # 实验数据
        self.experiment_data = {
            'start_time': datetime.now().isoformat(),
            'config': {},
            'metrics': {},
            'artifacts': []
        }

    def log_config(self, config):
        """记录配置"""
        self.experiment_data['config'] = config
        self.text_logger.log_config(config)

    def log_metrics(self, metrics, stage="training"):
        """记录指标"""
        if stage not in self.experiment_data['metrics']:
            self.experiment_data['metrics'][stage] = {}

        # 更新指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.experiment_data['metrics'][stage][key] = value
                # 记录到TensorBoard
                self.tb_logger.log_scalar(f"{stage}/{key}", value)

        # 记录到文本日志
        self.text_logger.log_metrics(metrics, stage)

    def log_artifact(self, artifact_type, artifact_path, description=""):
        """记录产出物（模型、图表等）"""
        artifact = {
            'type': artifact_type,
            'path': artifact_path,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment_data['artifacts'].append(artifact)
        self.text_logger.info(f"Artifact logged: {artifact_type} - {artifact_path}")

    def log_message(self, message, level="info"):
        """记录消息"""
        if level == "info":
            self.text_logger.info(message)
        elif level == "warning":
            self.text_logger.warning(message)
        elif level == "error":
            self.text_logger.error(message)

    def save_experiment(self):
        """保存实验数据"""
        self.experiment_data['end_time'] = datetime.now().isoformat()

        # 保存实验数据
        experiment_file = os.path.join(self.log_dir, "experiment.json")
        with open(experiment_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)

        # 关闭TensorBoard
        self.tb_logger.close()

        self.text_logger.info(f"Experiment data saved to: {experiment_file}")
        return experiment_file

    def get_logger(self):
        """获取文本记录器"""
        return self.text_logger

    def get_tb_logger(self):
        """获取TensorBoard记录器"""
        return self.tb_logger