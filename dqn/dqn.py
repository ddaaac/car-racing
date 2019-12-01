import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super().__init__()

        self.conv1_1 = nn.Conv2d(input_shape, 8, kernel_size=3, stride=1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(288, 256)
        self.linear2 = nn.Linear(256, num_of_actions)

    def forward(self, x):
        conv1_out = F.relu(self.batch_norm1(self.max_pool1(self.conv1_2(self.conv1_1(x)))))
        conv2_out = F.relu(self.batch_norm2(self.conv2_2(self.conv2_1(conv1_out))))
        conv3_out = F.relu(self.batch_norm3(self.conv3_2(self.conv3_1(conv2_out))))

        flattened = torch.flatten(conv3_out, start_dim=1)
        linear1_out = self.linear1(flattened)
        q_value = self.linear2(linear1_out)

        return q_value


if __name__ == '__main__':
    x = torch.rand(1, 1, 84, 84)
    dqn = DQN(input_shape=1, num_of_actions=4)
    a = dqn(x)
    m = a.max(1)[1]
    print(a)
    print(m)


