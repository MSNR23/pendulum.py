import gym
import numpy as np

# カスタム報酬関数を定義
def custom_reward(observation):
    # 速度が大きいほど報酬が大きくなるように設定
    velocity = observation[1]
    return velocity

# Q学習のエージェントクラスを定義
class QLearningAgent:
    def __init__(self, action_space_size, observation_space_size, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.2):
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Qテーブルの初期化
        self.q_table = np.zeros((observation_space_size, action_space_size))

    def select_action(self, state):
        # ε-greedy法により行動を選択
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.action_space_size)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        # Q値の更新
        current_q_value = self.q_table[state, action]
        max_future_q_value = np.max(self.q_table[next_state, :])
        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q_value)
        self.q_table[state, action] = new_q_value

# 環境の作成
env = gym.make('CartPole-v1')

# エージェントの作成
action_space_size = env.action_space.n
observation_space_size = env.observation_space.shape[0]
agent = QLearningAgent(action_space_size, observation_space_size)

# 学習の実行
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        # 行動の選択
        action = agent.select_action(state)

        # 行動の実行
        next_state, reward, done, _ = env.step(action)

        # カスタム報酬関数を適用
        reward = custom_reward(next_state)

        # Q値の更新
        agent.update_q_table(state, action, reward, next_state)

        # 次の状態への更新
        state = next_state

        # ゲーム終了時の処理
        if done:
            break

# 学習結果の確認
state = env.reset()
total_reward = 0

while True:
    env.render()
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

    if done:
        break

print("Total reward:", total_reward)

# 環境を閉じる
env.close()