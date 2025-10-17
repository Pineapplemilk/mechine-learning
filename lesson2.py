from math import log


def cal_shannon_ent(dataset):
    """
    计算熵
    """
    # 1. 计算数据集中样本的总数
    num_entries = len(dataset)   #表3.1中是5
    # 2. 创建一个字典，用于统计每个类别标签出现的次数
    labels_counts = {}
    # 3. 遍历数据集中的每条记录
    for feat_vec in dataset:   #[1, 1, 'yes']   [1,1, 'yes']  [1, 0, 'no']  [0, 1, 'no']  [0, 1, 'no']
        # feat_vec[-1] 表示每条样本的最后一个元素 类别标签
        current_label = feat_vec[-1]
        # 如果该标签是第一次出现，则在字典中初始化为 0
        # {}是字典，{key1:value1,key2:value2}
        if current_label not in labels_counts.keys(): 
            labels_counts[current_label] = 0
        # 累加该标签出现的次数
        labels_counts[current_label] += 1  #yse=2,NO=3

        print("类别统计：", labels_counts)  #YES=2,NO=3
    # 4. 计算香农熵
    shannon_ent = 0.0
    # 遍历字典中的每个类别及其计数
    for key in labels_counts:  #key有两个：yes和no
        # 计算该类别的概率
        prob = float(labels_counts[key])/num_entries   #3/5    计算yes出现的概率
        # 根据香农熵公式累加：
        shannon_ent -= prob*log(prob, 2)   
    # 5. 返回计算得到的熵值
    return shannon_ent


def create_dataSet():
    """
    熵接近 1，说明“yes”和“no”两个类别的比例比较接近，数据集的不确定性较高。
    熵接近 0,类别越集中，数据集越“纯”或“确定性越强”
    """
    dataset = [[1, 1, 'yes'],[1,1, 'yes'],[1, 0, 'no'],[0, 1, 'no'],[0, 1, 'no']]
    labels = ['no suerfacing', 'flippers']
    return dataset, labels


dataset, labels = create_dataSet()
#print(cal_shannon_ent(dataset))


def split_dataset(dataset, axis, value):
    """
    按照指定特征(axis)的某个取值(value)划分数据集。
    会选出所有该特征等于 value 的样本，
    并且返回时会去掉这一列特征。

    参数：
        dataset: 原始数据集（二维列表，每一行是一个样本，每一列是一个特征，最后一列通常是标签）
        axis: 要划分的特征列索引（例如 0 表示第 1 个特征）
        value: 特征的目标取值（例如 'sunny'）

    返回：
        ret_dataset: 划分后的子数据集（不包含 axis 那一列）
    """
    ret_dataset = []  # 用于存放划分后的子数据集
    # 遍历原始数据集的每一条样本
    for feat_vec in dataset:   #[0, 'sunny', 'yes'],    我想以天气情况来划分   axis=1,value=sunny
        # 如果这一条样本在 axis 特征上的值等于给定的 value
        if feat_vec[axis] == value:
            # 构建一个“去掉该特征”的新样本
            reduced_feat_vec = feat_vec[:axis]    # 取前面部分   reduced_feat_vec=[1]
            reduced_feat_vec.extend(feat_vec[axis+1:])  # 取后面部分拼接起来   feat_vec[axis+1:]=['yes']
            # 把这个新样本加入到子数据集中      #[1,'yes']
            ret_dataset.append(reduced_feat_vec)
      # 返回划分后的数据集
    return ret_dataset   #[[1,'yes'],[0,'yes']]


# 示例数据集：最后一列是标签是否出门，第一列是是否有风，第二列是天气情况
dataset_test = [
    [1, 'sunny', 'yes'],
    [1, 'rainy', 'no'],
    [0, 'sunny', 'yes']
]

# 按第0列的值为1来划分
result = split_dataset(dataset_test, 0, 1)   #[['sunny', 'yes']  ,  [ 'rainy', 'no']],
#print(result)


#dataset_test = [
#    [1, 'sunny', 'yes'],
#    [1, 'rainy', 'no'],
#    [0, 'sunny', 'yes']
##

#1.获取总的特征列
#2.拿每一个特征列，获取每个特征列的特征值
#3.使用该特征列，遍历每个特征值调用split，得到划分后的数据集
#4.计算该特征列的信息增益
#5.依次遍历计算所有特征列的信息增益，来比对哪个列的信息增益最大
#6.挑选信息增益最大的特征列作为主列


def choose_best_feature_split(dataset):
    """
    选择信息增益最大的特征索引，作为本轮划分的最优特征。

    参数：
        dataset: 数据集（二维列表，每行一条样本，最后一列是标签）
    返回：
        best_feature: 最优特征的索引位置
    """
    # 1. 计算特征总数（最后一列是标签，不算特征）
    num_features = len(dataset[0])-1    #2
    # 2. 计算原始数据集的熵（未划分前的不确定性）
    base_entropy = cal_shannon_ent(dataset)
    # 3. 初始化“最大信息增益”和“最佳特征”
    best_info_gain = 0.0
    best_feature = 1
    # 4. 遍历每一个特征，计算它的信息增益
    for i in range(num_features):    #num_features=2,range(2)=(0,1)
        # 4.1 提取出该特征所有样本的取值列表
        feat_list = [example[i] for example in dataset]    # [1, 'sunny', 'yes'] i=0  
        print(feat_list)
        #这是一个列表推导式的写法
        #等价于:
        #feat_list = []
        #for example in dataset:        # [1, 'sunny', 'yes'] i=1----sunny       [1, 'rainy', 'no'],------rainy   [0, 'sunny', 'yes']-----rainy   [sunny,rainy,sunny]
        #    feat_list.append(example[i])
        # 4.2 获取该特征的所有唯一取值,转换为set集合，自动去重
        unique_val = set(feat_list) #[sunny,rainy,sunny]
        print(unique_val)  #(sunny,rainy)
        # 4.3 计算该特征划分后的“加权平均熵”
        new_entropy = 0.0
        for value in unique_val:
            # 按照该特征的某个取值划分数据集
            sub_dataset = split_dataset(dataset, i, value)  #i=0的情况下，需要遍历2次，特征值有0，1
             # 计算该子集占整个数据集的比例
            prob = len(sub_dataset)/float(len(dataset))
            # 累加加权熵（概率 * 子集熵）
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        # 4.4 计算该特征的信息增益
        info_gain = base_entropy-new_entropy
        # 4.5 如果当前特征信息增益更大，就更新最优特征
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    # 5. 返回信息增益最大的特征索引
    return best_feature

loan_data = [
    
    #0列：收入水平:'高' '中' '低'
    #1列：是否有工作：'有工作' '无工作'
    #2列：信用评分：'良好' '一般' '差'
    #3列（标签列）： 贷款是否通过

    ['高', '有工作', '良好', 'yes'],
    ['高', '无工作', '一般', 'no'],
    ['中', '有工作', '良好', 'yes'],
    ['低', '有工作', '差', 'no'],
    ['低', '无工作', '一般', 'no'],
    ['高', '有工作', '差', 'yes'],
    ['中', '无工作', '良好', 'yes'],
    ['低', '有工作', '良好', 'yes']
]

print(choose_best_feature_split(loan_data))