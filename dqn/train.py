import torch
import gym
import torch.nn.functional as F
from torch.optim import RMSprop
from dqn.dqn import DQN
from dqn.actions import get_action_space, get_action
from dqn.replay_memory import ReplayMemory
from dqn.environment_wrapper import EnvironmentWrapper
from matplotlib import pyplot as plt
import csv
import time

class DQNTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_q_net = DQN(input_shape=1, num_of_actions=get_action_space())
        self.current_q_net.to(self.device)
        self.target_q_net = DQN(input_shape=1, num_of_actions=get_action_space())
        self.target_q_net.to(self.device)
        self.optimizer = RMSprop(self.current_q_net.parameters(),
                                 lr=self.params.lr)
        self.replay_memory = ReplayMemory(self.params.memory_capacity)
        env = gym.make('CarRacing-v0')
        self.environment = EnvironmentWrapper(env, self.params.skip_steps)

        self.max_neg_step = 100
        self.neg_reward_cnt = 0
        self.punishment = -20.0

    def run(self):
        state = torch.tensor(self.environment.reset(),
                             device=self.device,
                             dtype=torch.float32)
        self._update_target_q_net()
        plot = [[], []]
        time_arr = []
        total_reward = 0
        episode = 1
        plt.figure()
        start_time = time.time()
        last_done_time = time.time()
        for step in range(int(self.params.num_of_steps)):
            q_value = self.current_q_net(torch.stack([state]))
            action_index, action = get_action(q_value,
                                              train=True,
                                              step=step,
                                              params=self.params,
                                              device=self.device)
            next_state, reward, done = self.environment.step(action)
            next_state = torch.tensor(next_state,
                                      device=self.device,
                                      dtype=torch.float32)
            self.replay_memory.add(state, action_index, reward, next_state, done)
            state = next_state


            if reward < 0:
                self.neg_reward_cnt += 1
            else:
                self.neg_reward_cnt = 0


            total_reward += reward


            if self.neg_reward_cnt > self.max_neg_step:
                if total_reward < 500:
                    total_reward += self.punishment
                done = True



            if done:
                state = torch.tensor(self.environment.reset(),
                                     device=self.device,
                                     dtype=torch.float32)
                plot[0].append(episode)
                plot[1].append(total_reward)
                episode += 1
                plt.plot(plot[0], plot[1])
                time_arr.append(last_done_time-time.time())
                last_done_time = time.time()
                plt.savefig('../drive/My Drive/reward_plot_pooling.png')
                csv_file = open("../drive/My Drive/reward_csv_pooling.csv", "w", newline="\n")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(plot[1])
                csv_writer.writerow(time_arr)
                csv_file.close()
                print("An episode is over. Reward: {}".format(total_reward))

                self.neg_reward_cnt = 0

                total_reward = 0
            if len(self.replay_memory.memory) > self.params.batch_size:
                loss = self._update_current_q_net()
            if step % self.params.target_update_freq == 0:
                self._update_target_q_net()
        torch.save(self.target_q_net.state_dict(), self.model_path)
        print("training Runtime : {} min".format((time.time() - start_time)/60))

    def _update_current_q_net(self):
        batch = self.replay_memory.sample(self.params.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        q_values = self.current_q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1)[0]

        expected_q_values = rewards + self.params.discount_factor * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _update_target_q_net(self):
        self.target_q_net.load_state_dict(self.current_q_net.state_dict())
