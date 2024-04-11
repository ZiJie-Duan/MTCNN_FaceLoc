
"""
此文件用于分割数据集 到 三个部分：训练集、验证集、测试集
切割CSV文件
"""

import csv


def split_dataset(csv_file, proportion=[]):
    # 读取CSV文件 并切割, portion = [train, val, test] 每个部分的比例
    # 将数据集切割成三个部分：训练集、验证集、测试集 并保存到三个文件中

    print('Splitting dataset...')
    print('CSV file: ', csv_file)
    print('Proportion: ', proportion)
    
    data = None
    # 读取CSV文件
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    # 计算每个部分的数量
    total = len(data)
    train_num = int(total * proportion[0])
    val_num = int(total * proportion[1])
    test_num = total - train_num - val_num

    # 切割数据集
    train_data = data[:train_num]
    val_data = data[train_num:train_num + val_num]
    test_data = data[train_num + val_num:]

    # 保存到文件
    train_file = csv_file.replace('.csv', '_train.csv')
    val_file = csv_file.replace('.csv', '_val.csv')
    test_file = csv_file.replace('.csv', '_test.csv')

    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_data)
    
    with open(val_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_data)

    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)
    
    return None


if __name__ == '__main__':
    # 测试
    split_dataset(r"C:\Users\lucyc\Desktop\face_loc_dataset.csv", [0.6, 0.05, 0.35])
    print('Done!')



