from torch import nn


class DDQN(nn.Module):
    def __init__(self, observation_shape, action_size, use_conv=True):
        super().__init__()

        # Because frames (4) frames are stacked we use them as if they were channels
        self.rolling_frames = observation_shape[0] if use_conv else observation_shape
        self.action_size = action_size
        self.update_count = 0
        self.target_net_update_freq = 1000

        self.conv = nn.Sequential(
            nn.Conv2d(self.rolling_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten()
        ) if use_conv else nn.Sequential()

        self.fc = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(self.action_size)
        )
        self.target_Q_net = nn.Sequential(self.conv, self.fc)
        self.Q_net = nn.Sequential(self.conv, self.fc)

    def update_target(self):
        if self.update_count % self.target_net_update_freq == 0:
            self.target_Q_net.load_state_dict(self.Q_net.state_dict())
        self.update_count += 1

    def forward(self, state):
        return self.Q_net(state)
