import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
"""
小试牛刀
"""


# 定义策略网络
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


# 定义值函数网络
class ValueFunction(nn.Module):
    def __init__(self, input_size):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义PPO算法
class PPO:
    def __init__(self, input_size, output_size, epsilon_clip, c1, c2, lr):
        self.policy = Policy(input_size, output_size)
        self.value_function = ValueFunction(input_size)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_function.parameters(), lr=lr)
        self.epsilon_clip = epsilon_clip
        self.c1 = c1
        self.c2 = c2

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        """
        torch.from_numpy()函数将NumPy数组转换为PyTorch张量。
        .float()方法将张量的数据类型转换为浮点型。
        .unsqueeze(0)方法在张量的维度0上添加一个额外的维度，这样张量的形状变为(1, ...)
        """
        probs = self.policy(state)# 输出在每个动作上的概率
        m = Categorical(probs)# 创建了一个Categorical对象m，它是一个带有给定概率列表probs的多项式分布。
        action = m.sample()# action.item()是一个用于提取PyTorch张量中单个标量值的方法
        return action.item(), m.log_prob(action)#获取采样类别的对数概率

    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # 更新值函数网络
        values = self.value_function(states)# values.squeeze()多行转成一行
        value_loss = F.mse_loss(values.squeeze(), returns)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # 更新策略网络
        new_probs = self.policy(states)
        new_dist = Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value = self.value_function(state)
        return value.item()

    def save(self):
        torch.save(self.policy.state_dict(), 'PPO_policy_dec.pth')
        # torch.save(self.actor_old.state_dict(), 'actor_old.pth')
        torch.save(self.value_function.state_dict(), 'PPO_value_function_dec.pth')
        print('...save model...')

    def load(self):

        self.policy.load_state_dict(torch.load('PPO_policy_dec.pth'))
        # self.actor_old.load_state_dict(torch.load('actor_old.pth'))
        self.value_function.load_state_dict(torch.load('PPO_value_function_dec.pth'))
        print('...load...')

# 创建CartPole环境
env = gym.make('CartPole-v0')

# 设置超参数
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
epsilon_clip = 0.2
c1 = 1.0
c2 = 0.01
lr = 0.001
epochs = 1000
max_steps = 200
gamma = 0.99
Batch = 64
train = False
test = True
# 创建PPO对象
agent = PPO(input_size, output_size, epsilon_clip, c1, c2, lr)
if train == True:
# 训练PPO算法
    for epoch in range(epochs):
        state = env.reset()[0]
        done = False
        total_reward = 0
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(agent.get_value(state))

            state = next_state
            total_reward += reward

            # if done:
            #     break
            if (step + 1) % Batch == 0 or done == True :
                returns = []
                advantages = []
                R = 0

                for t in reversed(range(len(rewards))):# 计算折扣奖励和优势函数
                    R = rewards[t] + gamma * R
                    returns.insert(0, R) # 将一个值 R 插入到列表 returns 的开头。
                    advantages.insert(0, R - values[t])

                agent.update(states, actions, log_probs, returns, advantages)


        if epoch % 10 == 0:
            print('Epoch: {}, Total Reward: {}'.format(epoch, total_reward))
    agent.save()
    env.close()
"""
returns = [] 和 advantages = []：初始化回报值和优势值的空列表。
R = 0：初始化累积回报值R为0。
for t in reversed(range(len(rewards))):：从最后一个时间步开始迭代，直到第一个时间步。
R = rewards[t] + gamma * R：计算累积回报值R，通过将当前时间步的奖励值与之前的累积回报值乘以折扣因子相加。
returns.insert(0, R)：将计算得到的回报值R插入到列表returns的最前面，以保持时间顺序。
advantages.insert(0, R - values[t])：计算并将优势值(R - values[t])插入到列表advantages的最前面，其中values[t]是对应时间步的价值函数预测值
"""
if test == True:
    agent.load()
    for episode in range(10):
        s = env.reset()[0]
        done = False
        ep_r = 0
        while not done:
            env.render()
            a, _ = agent.select_action(s)
            s_, rew, done, _,_ = env.step(a)
            ep_r += rew
            s = s_

        print(f"episode:{episode} ep_r:{ep_r}")

