import os
import csv
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义线条样式组合（颜色+线条），用于区分不同实验，可根据需要扩展
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


def load_tensorboard_data(log_dir):
    """加载单个TensorBoard日志文件中的标量数据（损失函数）"""
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        raise ValueError(f"在目录 {log_dir} 中未找到TensorBoard事件文件")
    
    # 按修改时间排序，取最新的事件文件
    event_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    latest_event = os.path.join(log_dir, event_files[0])
    
    ea = event_accumulator.EventAccumulator(
        latest_event,
        size_guidance={event_accumulator.SCALARS: 0}  # 加载所有标量数据
    )
    ea.Reload()
    
    # 提取所有包含"loss"的标量（不区分大小写）
    data = {}
    for tag in ea.Tags()['scalars']:
        if "loss" in tag.lower():
            events = ea.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            data[tag] = {"steps": steps, "values": values}
    
    return data


def save_to_csv(all_data, csv_root_dir):
    """
    将所有实验的损失曲线数据保存为CSV文件
    
    结构：
    - 根目录：csv_root_dir
      - 实验1文件夹（以标签命名）
        - 损失函数1.csv
        - 损失函数2.csv
      - 实验2文件夹
        - 损失函数1.csv
        - ...
    """
    os.makedirs(csv_root_dir, exist_ok=True)  # 创建根目录
    
    for exp in all_data:
        exp_label = exp["label"]
        exp_data = exp["data"]
        
        # 处理标签中的特殊字符（避免文件夹命名错误）
        safe_label = exp_label.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
        exp_dir = os.path.join(csv_root_dir, safe_label)
        os.makedirs(exp_dir, exist_ok=True)  # 创建当前实验的文件夹
        
        # 为每个损失函数保存CSV
        for loss_name, loss_data in exp_data.items():
            # 处理损失函数名称中的特殊字符（避免文件名错误）
            safe_loss_name = loss_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
            csv_path = os.path.join(exp_dir, f"{safe_loss_name}.csv")
            
            # 写入CSV（格式：step,value）
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "value"])  # 表头
                # 写入每一行数据
                for step, value in zip(loss_data["steps"], loss_data["values"]):
                    writer.writerow([step, value])
        
        print(f"已保存实验 '{exp_label}' 的CSV文件至：{exp_dir}")


def plot_loss_curves(all_data, save_dir=None):
    """绘制多个实验中同名损失函数的对比曲线（保持原有绘图功能）"""
    all_loss_names = set()
    for item in all_data:
        all_loss_names.update(item["data"].keys())
    all_loss_names = sorted(all_loss_names)
    
    for loss_name in all_loss_names:
        plt.figure(figsize=(10, 6))
        
        for i, exp in enumerate(all_data):
            exp_label = exp["label"]
            exp_data = exp["data"]
            
            if loss_name in exp_data:
                color = COLORS[i % len(COLORS)]
                linestyle = LINESTYLES[i % len(LINESTYLES)]
                plt.plot(
                    exp_data[loss_name]["steps"],
                    exp_data[loss_name]["values"],
                    label=exp_label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2
                )
        
        plt.title(f"{loss_name} 收敛曲线对比", fontsize=12)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Loss", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=9)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_loss_name = loss_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            plt.savefig(
                os.path.join(save_dir, f"{safe_loss_name}_comparison.png"),
                dpi=300,
                bbox_inches='tight'
            )
        
        # plt.show()
        
def main():
    # 解析命令行参数（支持任意多个日志目录和标签）
    parser = argparse.ArgumentParser(description='对比多个TensorBoard日志中的损失函数收敛曲线')
    parser.add_argument('--log_dirs', default=['kidney3-PMAR2-new2-discri5iter', 
                                               'kidney3-PMAR2-new2-discri5iter-woProto',
                                               'kidney3-PMAR2-new2-autoSelectProto',
                                               'kidney3-PMAR2-new2-discri5iter-woModal',
                                               'kidney3-PMAR2-new2-discri5iter-woMorp',
                                               'kidney3-PMAR2-new2-discri5iter-NCC',], nargs='+', help='多个TensorBoard日志目录（空格分隔）')
    parser.add_argument('--labels', default=['All Components','W/o Proto','AutoSelect','W/o Modal','W/o Morph','Using NCC'],nargs='*', help='每个日志目录对应的标签（可选，默认自动生成）')
    parser.add_argument('--save-dir', default='save_loss_curve', help='图像保存目录（可选）')
    parser.add_argument('--csv-dir', default='save_loss_curve', help='CSV文件保存根目录（可选，指定则保存）')  # 新增CSV保存参数
    
    args = parser.parse_args()
    
    # 处理标签
    num_experiments = len(args.log_dirs)
    if not args.labels or len(args.labels) < num_experiments:
        default_labels = [f"实验{i+1}" for i in range(num_experiments)]
        if args.labels:
            default_labels[:len(args.labels)] = args.labels
        args.labels = default_labels
    
    try:
        # 加载所有实验数据
        all_data = []
        for i, log_dir in enumerate(args.log_dirs):
            print(f"正在加载第{i+1}/{num_experiments}个日志: {log_dir}...")
            data = load_tensorboard_data(log_dir)
            all_data.append({
                "label": args.labels[i],
                "data": data
            })
        
        # 检查是否有损失数据
        if not any(exp["data"] for exp in all_data):
            print("警告: 所有日志文件中均未找到损失函数数据")
            return
        
        # 保存CSV文件（如果指定了目录）
        if args.csv_dir:
            print("开始保存CSV文件...")
            save_to_csv(all_data, args.csv_dir)
        
        # 绘制曲线（保持原有功能）
        print("开始绘制损失曲线...")
        plot_loss_curves(all_data, args.save_dir)
        
        print("处理完成！")
        if args.save_dir:
            print(f"图像已保存至：{args.save_dir}")
        if args.csv_dir:
            print(f"CSV文件已保存至：{args.csv_dir}")
    
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()