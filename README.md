# weiboClassifier
python实现朴素贝叶斯分类器


### build1 
#### date:2017-7-10

- 实现三条数值特征的提取，先验概率通过计算高斯概率密度函数的方法得到
- 实现对单条微博进行检测，输出检测结果是否正确
- 训练集随机生成，为总数据集的2/3

### build2
#### date:2017-7-11

- 增加了数据中提取的特征，有4个数值连续的特征和3个数值离散的特征
- 实现对测试集数据逐条进行检测分类，输出每一条的检测结果和最后的检测正确率
- 训练集和测试集通过已有数据集进行随机划分

#### 已提取特征

- 用户被关注度
- 用户互粉度
- 发布微博频率
- 用户收藏数
- 是否允许所有人私信
- 是否开启定位功能
- 是否有个人简介
