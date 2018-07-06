
# coding: utf-8

# # 你的第一个神经网络
# 
# 在此项目中，你将构建你的第一个神经网络，并用该网络预测每日自行车租客人数。我们提供了一些代码，但是需要你来实现神经网络（大部分内容）。提交此项目后，欢迎进一步探索该数据和模型。

# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 加载和准备数据
# 
# 构建神经网络的关键一步是正确地准备数据。不同尺度级别的变量使网络难以高效地掌握正确的权重。我们在下方已经提供了加载和准备数据的代码。你很快将进一步学习这些代码！

# In[26]:


data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)


# In[27]:


rides.head()


# ## 数据简介
# 
# 此数据集包含的是从 2011 年 1 月 1 日到 2012 年 12 月 31 日期间每天每小时的骑车人数。骑车用户分成临时用户和注册用户，cnt 列是骑车用户数汇总列。你可以在上方看到前几行数据。
# 
# 下图展示的是数据集中前 10 天左右的骑车人数（某些天不一定是 24 个条目，所以不是精确的 10 天）。你可以在这里看到每小时租金。这些数据很复杂！周末的骑行人数少些，工作日上下班期间是骑行高峰期。我们还可以从上方的数据中看到温度、湿度和风速信息，所有这些信息都会影响骑行人数。你需要用你的模型展示所有这些数据。

# In[28]:


rides[:24*10].plot(x='dteday', y='cnt')


# ## 额外的数据探索与可视化分析(EDA)
# 根据上述的提示，我针对所有特征进行了额外的分析，以期加深对特征和数据集的理解，挖掘数据中潜在的规律，并分析归纳出一些结论。  
# 
# 这里主要以平均骑行人数和总平均骑行人数作为衡量标准，分类探索各个特征对临时`casual`、注册`registered`及总骑行人数`cnt`的影响，特征分析包括：
# * 星期分布`weekday`、时间分布`hr`-平均骑行人数
# * 假期`holiday`和工作日`holiday`与否-平均骑行人数
# * 温度`temp`、体感温度`atemp`、湿度`hum`和风速`windspeed`-平均骑行人数、总骑行人数
# * 季节`season`和天气`weathersit`-平均骑行人数
# * 年份`yr`、日期`dteday`-总骑行人数
# 
# 在特征分布均匀的情况下，均值和总值实际是等效的（如`weekday`, `season`, `hr`, `dteday`）。  
# 在特征分布不均的情况下，均值作为主要衡量指标，总值需要作为辅助判断，原因是对于特征样本较少的部分，可能会出现少量噪声数据或特殊事件数据对分析结果产生较大的干扰。对于样本较少的部分，在分析时会降低其可信度，因为其分布可能来源于其他特征的影响。

# In[29]:


# matplot绘制辅助函数
# 绘制单个分布图，参数包括dataframe、特征、目标、显示类型、均值/总值、轴显示、数据缩放
def plot_distribution(data, feature, targets=['casual', 'registered', 'cnt'], plot_kind='line', plot_mean=True, xlabels=[], scale=1):
    if plot_mean:
        group_data = data.groupby(feature)[targets].mean()
    else:
        group_data = data.groupby(feature)[targets].sum()
    group_data.reset_index(inplace=True)
    
    if scale != 1:
        group_data[feature] *= scale

    ax = group_data.plot(x=feature, y=targets, kind=plot_kind, figsize=(8, 5),
                         title='distribution of %s - %s' % (feature, 'mean' if plot_mean else 'total'))
    ax.set_ylabel('%s count' % 'mean' if plot_mean else 'total')

    if xlabels:
        ax.axes.set_xticklabels(xlabels, rotation=0)
    
# 绘制多个分布图，参数包括dataframe、特征、目标、显示类型、均值/总值、轴显示、数据缩放
def plot_distributions(data, features, targets=['casual', 'registered', 'cnt'], plot_kinds=[], plot_means=[], xlabels=[], scales=[]):
    fig, axes = plt.subplots(nrows=1, ncols=len(features))
        
    for i, feature in enumerate(features):
        if plot_means[i]:
            group_data = data.groupby(feature)[targets].mean()
        else:
            group_data = data.groupby(feature)[targets].sum()
        group_data.reset_index(inplace=True)
        
        if scales:
            group_data[feature] *= scales[i]
        ax = group_data.plot(x=feature, y=targets, kind=plot_kinds[i], figsize=(15, 5), ax=axes[i], 
                             title='distribution of %s - %s' % (feature, 'mean' if plot_means[i] else 'total'))
        ax.set_ylabel('%s count' % 'mean' if plot_means[i] else 'total')
        
        if xlabels:
            ax.axes.set_xticklabels(xlabels[i], rotation=0)
        
    plt.show()


# ### 星期分布`weekday`、时间分布`hr`-平均骑行人数
# * 注册用户所占比例和使用频率远高于临时用户，因此总用户的分布基本与注册用户类似，因此我们主要关注注册用户和临时用户的分布差异。  
# * 结合星期和时间分布可以看出，注册用户偏向于周一到周五上下班高峰期出行，可以推测出单车主要用于上下班或上下课，注册用户主要是上班族和学生。  
# * 而临时用户则喜好在周末和错峰出行，可以推测用户群主要是家庭主妇，或者是上下班使用其他交通工具，而选择周末骑行游玩为主的用户。由于使用频率较低，所以选择不注册，跟其特征稳合。

# In[30]:


plot_distributions(rides, ['weekday', 'weekday'], plot_kinds=['line', 'bar'], plot_means=[True, True],
                  xlabels=[['', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
                           ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']])
plot_distribution(rides, 'hr')


# ### 假期`holiday`和工作日`holiday`与否-平均骑行人数
# * 对于假期和工作日，注册用户和临时用户的趋势也是相反的。  
# * 注册用户偏向于在工作日和非假期使用共享单车，而临时用户在假期和非工作日使用频率较高，进一步验证了之前的想法。

# In[31]:


plot_distributions(rides, ['holiday', 'workingday'], plot_kinds=['bar', 'bar'], plot_means=[True, True],
                  xlabels=[['no', 'yes'], 
                           ['no', 'yes']])


# ### 温度`temp`、体感温度`atemp`、湿度`hum`和风速`windspeed`-平均骑行人数、总骑行人数
# * 为了方便分析数据，这里还原了真实的温度、体感温度、湿度和风速（原数据集已做归一化处理）
# * 前面已经提到过，对于分布不均匀的特征，以均值为主要衡量标准，总值需要作为辅助判断，降低样本极少区域的分布可信度。如`temp`, `atemp`, `windspeed`的右边界区域，sum几乎都趋近于0，而三者的均值都有一定程度的反常，而这部分我们是可以忽略掉的。
# * 对于曲线中的剧烈波动，这些低谷的产生可能是由于分布不连续造成的，真实数据无相应的采样样本，可以不予考虑。
# * 主要关注曲线趋势，横向比较，我们可以看出，对于这四方面因素，不同用户群的喜好一致，都倾向于在舒适度较高的条件下骑行。
# * 唯一反常的点是，当湿度`hum`位于[80, 95]的高湿度区间下，骑行总人数反而增多。仔细分析一下，其实是因为当地下雨情况较多，产生更多的样本，其实平均骑行人数是在减少的，同样满足上述规则。

# In[32]:


plot_distributions(rides, ['temp', 'temp'], plot_kinds=['line', 'line'], plot_means=[True, False], scales=[41, 41])
plot_distributions(rides, ['atemp', 'atemp'], plot_kinds=['line', 'line'], plot_means=[True, False], scales=[50, 50])
plot_distributions(rides, ['hum', 'hum'], plot_kinds=['line', 'line'], plot_means=[True, False], scales=[100, 100])
plot_distributions(rides, ['windspeed', 'windspeed'], plot_kinds=['line', 'line'], plot_means=[True, False], scales=[67, 67])


# ### 季节`season`和天气`weathersit`-平均骑行人数
# * 对于季节和天气方面的影响，不同用户群也是一致的。
# * 与温湿度风速一致，用户也是倾向于在人体舒适度较高的时候骑行。
# * 至于为什么春季骑行的人数较少，分析数据可以看出，春季是12月-3月，并非中国农历传统意义上的春季，受到低温和节日的影响。

# In[33]:


plot_distributions(rides, ['season', 'weathersit'], plot_kinds=['bar', 'bar'], plot_means=[True, True], 
                   xlabels=[['spring', 'summer', 'autumn', 'winter'], ['sunny', 'cloudy', 'rainy', 'rain heavily']])


# ### 年份`yr`、日期`dteday`-总骑行人数
# 从年份和日期分布可以看出，随着时间的推移，总骑行人数都是递增的，说明这款单车的普及率和使用率在逐步提升。

# In[34]:


plot_distributions(rides, ['yr', 'yr'], plot_kinds=['bar', 'pie'], plot_means=[False, False], xlabels=[['2011', '2012'],['2011', '2012']])
plot_distribution(rides, 'dteday', plot_mean=False)


# ### 虚拟变量（哑变量）
# 
# 下面是一些分类变量，例如季节、天气、月份。要在我们的模型中包含这些数据，我们需要创建二进制虚拟变量。用 Pandas 库中的 `get_dummies()` 就可以轻松实现。

# In[35]:


dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()


# ### 调整目标变量
# 
# 为了更轻松地训练网络，我们将对每个连续变量标准化，即转换和调整变量，使它们的均值为 0，标准差为 1。
# 
# 我们会保存换算因子，以便当我们使用网络进行预测时可以还原数据。

# In[36]:


quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# ### 将数据拆分为训练、测试和验证数据集
# 
# 我们将大约最后 21 天的数据保存为测试数据集，这些数据集会在训练完网络后使用。我们将使用该数据集进行预测，并与实际的骑行人数进行对比。

# In[37]:


# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


# 我们将数据拆分为两个数据集，一个用作训练，一个在网络训练完后用来验证网络。因为数据是有时间序列特性的，所以我们用历史数据进行训练，然后尝试预测未来数据（验证数据集）。

# In[38]:


# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# ## 开始构建网络
# 
# 下面你将构建自己的网络。我们已经构建好结构和反向传递部分。你将实现网络的前向传递部分。还需要设置超参数：学习速率、隐藏单元的数量，以及训练传递数量。
# 
# <img src="assets/neural_network.png" width=300px>
# 
# 该网络有两个层级，一个隐藏层和一个输出层。隐藏层级将使用 S 型函数作为激活函数。输出层只有一个节点，用于递归，节点的输出和节点的输入相同。即激活函数是 $f(x)=x$。这种函数获得输入信号，并生成输出信号，但是会考虑阈值，称为激活函数。我们完成网络的每个层级，并计算每个神经元的输出。一个层级的所有输出变成下一层级神经元的输入。这一流程叫做前向传播（forward propagation）。
# 
# 我们在神经网络中使用权重将信号从输入层传播到输出层。我们还使用权重将错误从输出层传播回网络，以便更新权重。这叫做反向传播（backpropagation）。
# 
# > **提示**：你需要为反向传播实现计算输出激活函数 ($f(x) = x$) 的导数。如果你不熟悉微积分，其实该函数就等同于等式 $y = x$。该等式的斜率是多少？也就是导数 $f(x)$。
# 
# 
# 你需要完成以下任务：
# 
# 1. 实现 S 型激活函数。将 `__init__` 中的 `self.activation_function`  设为你的 S 型函数。
# 2. 在 `train` 方法中实现前向传递。
# 3. 在 `train` 方法中实现反向传播算法，包括计算输出错误。
# 4. 在 `run` 方法中实现前向传递。
# 
#   

# In[53]:


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
            
            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
            final_outputs = final_inputs # signals from final output layer
            
            #### Implement the backward pass here ####
            ### Backward pass ###
            
            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs # Output layer error is the difference between desired target and actual output.
            
            # 这里替换了output_error_term和hidden_error的赋值顺序
            # 根据反向传播公式，隐藏层的误差项应由输出层的误差项计算得出，虽然在这里输出层误差值与误差项相等
            output_error_term = error
            
            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
            
            # TODO: Backpropagated error terms - Replace these values with your calculations.
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
            
            # Weight step (hidden to output)
            delta_weights_h_o += hidden_outputs[:, None] * output_error_term
            # Weight step (input to hidden)
            delta_weights_i_h += X[:, None] * hidden_error_term

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
 
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


# In[54]:


def MSE(y, Y):
    return np.mean((y-Y)**2)


# ## 单元测试
# 
# 运行这些单元测试，检查你的网络实现是否正确。这样可以帮助你确保网络已正确实现，然后再开始训练网络。这些测试必须成功才能通过此项目。

# In[55]:


import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328], 
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)


# ## 训练网络
# 
# 现在你将设置网络的超参数。策略是设置的超参数使训练集上的错误很小但是数据不会过拟合。如果网络训练时间太长，或者有太多的隐藏节点，可能就会过于针对特定训练集，无法泛化到验证数据集。即当训练集的损失降低时，验证集的损失将开始增大。
# 
# 你还将采用随机梯度下降 (SGD) 方法训练网络。对于每次训练，都获取随机样本数据，而不是整个数据集。与普通梯度下降相比，训练次数要更多，但是每次时间更短。这样的话，网络训练效率更高。稍后你将详细了解 SGD。
# 
# 
# ### 选择迭代次数
# 
# 也就是训练网络时从训练数据中抽样的批次数量。迭代次数越多，模型就与数据越拟合。但是，如果迭代次数太多，模型就无法很好地泛化到其他数据，这叫做过拟合。你需要选择一个使训练损失很低并且验证损失保持中等水平的数字。当你开始过拟合时，你会发现训练损失继续下降，但是验证损失开始上升。
# 
# ### 选择学习速率
# 
# 速率可以调整权重更新幅度。如果速率太大，权重就会太大，导致网络无法与数据相拟合。建议从 0.1 开始。如果网络在与数据拟合时遇到问题，尝试降低学习速率。注意，学习速率越低，权重更新的步长就越小，神经网络收敛的时间就越长。
# 
# 
# ### 选择隐藏节点数量
# 
# 隐藏节点越多，模型的预测结果就越准确。尝试不同的隐藏节点的数量，看看对性能有何影响。你可以查看损失字典，寻找网络性能指标。如果隐藏单元的数量太少，那么模型就没有足够的空间进行学习，如果太多，则学习方向就有太多的选择。选择隐藏单元数量的技巧在于找到合适的平衡点。

# In[71]:


import sys

### TODO:Set the hyperparameters here, you need to change the defalut to get a better solution ###
iterations = 100000
learning_rate = 0.05
hidden_nodes = 35
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations))                      + "% ... Training loss: " + str(train_loss)[:5]                      + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# In[74]:


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()


# ## 检查预测结果
# 
# 使用测试数据看看网络对数据建模的效果如何。如果完全错了，请确保网络中的每步都正确实现。

# In[75]:


fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# ## 可选：思考下你的结果（我们不会评估这道题的答案）
# 
#  
# 请针对你的结果回答以下问题。模型对数据的预测效果如何？哪里出现问题了？为何出现问题呢？
# 
# > **注意**：你可以通过双击该单元编辑文本。如果想要预览文本，请按 Control + Enter
# 
# #### 请将你的答案填写在下方
# 在模型训练之前，我首先对数据集进行了额外的数据探索与可视化分析（EDA），以期加深对特征和数据集的理解，挖掘数据中潜在的规律，并分析归纳出一些结论。但由于没有特征工程的阶段，其实对模型优化并没有起到太大的帮助，只能作为特征分析方面的练手。
# 
# 
# 默认的超参数对数据的预测效果一般，主要原因是隐藏单元数目太少，迭代次数不够，学习速度较大，导致预测整体精度不够，欠拟合较严重。
# 
# 
# 对于超参数的调节，我的思路如下：
# * 隐藏单元数量：我首先将隐藏单元数量扩大到200、400，发现隐藏单元数目过多，导致学习方向有太多的选择，误差常常会回调，梯度下降得很慢。后面将隐藏单元下调到[30-40]，梯度下降速度和模型精度都有所提升
# * 学习速率：为提升精度，学习率尝试下调到0.01、0.02，收敛速度过慢，需要太多的迭代次数，后上调到0.05，以权衡收敛速度和拟合精度
# * 迭代次数：依据早期停止原则，迭代次数可以通过观察epoch-error图，validation_loss停止降低并开始增大时作为迭代次数的最佳值，防止过拟合。
# 
# 以下是手工调整超参数进行模型训练的一些不完全统计(每次运行结果会有所出入)：
# 
# | iterations | learning_rate | hidden_nodes | training_loss | validation_loss |
# | - | - | - | - | - | 
# | 10000 | 0.01 | 200 | 0.304 | 0.472 |
# | 20000 | 0.02 | 400 | 0.279 | 0.445 |
# | 20000 | 0.02 | 40 | 0.227 | 0.401 |
# | 20000 | 0.01 | 35 | 0.290 | 0.456 |
# | 20000 | 0.05 | 30 | 0.219 | 0.379 |
# | 40000 | 0.03 | 35 | 0.134 | 0.262 |
# | 30000 | 0.05 | 30 | 0.088 | 0.177 |
# | 35000 | 0.05 | 35 | 0.081 | 0.196 |
# | 40000 | 0.05 | 30 | 0.074 | 0.169 |
# | 40000 | 0.05 | 35 | 0.075 | 0.169 |
# | 40000 | 0.05 | 40 | 0.083 | 0.175 |
# | 50000 | 0.05 | 35 | 0.061 | 0.165 |
# | 60000 | 0.05 | 30 | 0.061 | 0.176 |
# | 60000 | 0.05 | 35 | 0.058 | 0.136 |
# | 100000 | 0.05 | 35 | 0.048 | 0.129 |
# 
# 
# 根据以上思路，确定超参数的大致范围，然后再进行精调，最终确定的最佳参数为：  
# * iterations = 100000
# * learning_rate = 0.05
# * hidden_nodes = 35
# * output_nodes = 1
# 
# 达到精度要求：training loss< 0.09 and validation loss < 0.18
# 
