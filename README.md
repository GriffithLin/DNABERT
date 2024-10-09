# 准备
输入数据tsv格式和环境配置详见DNABERT
# 运行
使用examples目录下的exert_feature.py进行DNA序列的特征提取
## 具体设置：
args.data_dir 数据的目录
args.data_name_list 要输入的数据文件组成的列表
args.predict_dir  输出特征目录
args.output_name_list 对应data_name_list的输出文件列表。
crop_len_list 序列截取长度，方便设置文件目录 和 args.max_seq_length
k_mer_list  k—mer





