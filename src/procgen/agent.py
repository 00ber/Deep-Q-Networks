import torch 
import numpy as np
import random
import torch.nn as nn
import copy
import time, datetime
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class DQNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        print("#################################")
        print("#################################")
        print(input_dim)
        print(output_dim)
        print("#################################")
        print("#################################")
        c, h, w = input_dim
        
        # if h != 84:
        #     raise ValueError(f"Expecting input height: 84, got: {h}")
        # if w != 84:
        #     raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7168, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)



class MetricLogger:
    def __init__(self, save_dir):
        self.writer = SummaryWriter(log_dir=save_dir)
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self, episode_number):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.writer.add_scalar("Avg Loss for episode", ep_avg_loss, episode_number)
        self.writer.add_scalar("Avg Q value for episode", ep_avg_q, episode_number)
        self.writer.flush()
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        self.writer.add_scalar("Mean reward last 100 episodes", mean_ep_reward, episode)
        self.writer.add_scalar("Mean length last 100 episodes", mean_ep_length, episode)
        self.writer.add_scalar("Mean loss last 100 episodes", mean_ep_loss, episode)
        self.writer.add_scalar("Mean reward last 100 episodes", mean_ep_reward, episode)
        self.writer.add_scalar("Epsilon value", epsilon, episode)
        self.writer.add_scalar("Mean Q Value last 100 episodes", mean_ep_q, episode)
        self.writer.flush()
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


class DQNAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 save_dir, 
                 checkpoint=None, 
                 learning_rate=0.00025, 
                 max_memory_size=100000, 
                 batch_size=32,
                 exploration_rate=1,
                 exploration_rate_decay=0.9999999,
                 exploration_rate_min=0.1,
                 training_frequency=1,
                 learning_starts=1000,
                 target_network_sync_frequency=500,
                 reset_exploration_rate=False, 
                 save_frequency=100000,
                 gamma=0.9,
                 load_replay_buffer=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_memory_size = max_memory_size
        self.memory = deque(maxlen=max_memory_size)
        self.batch_size = batch_size

        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.gamma = gamma

        self.curr_step = 0
        self.learning_starts = learning_starts  # min. experiences before training

        self.training_frequency = training_frequency   # no. of experiences between updates to Q_online
        self.target_network_sync_frequency = target_network_sync_frequency  # no. of experiences between Q_target & Q_online sync

        self.save_every = save_frequency   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = DQNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint, reset_exploration_rate, load_replay_buffer)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, amsgrad=True)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    # def td_estimate(self, state, action):
    #     current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
    #     return current_Q


    # @torch.no_grad()
    # def td_target(self, reward, next_state, done):
    #     next_state_Q = self.net(next_state, model='online')
    #     best_action = torch.argmax(next_state_Q, axis=1)
    #     next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
    #     return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def td_estimate(self, states, actions):
        actions = actions.reshape(-1, 1)
        predicted_qs = self.net(states, model='online')# Q_online(s,a)
        predicted_qs = predicted_qs.gather(1, actions)
        return predicted_qs


    @torch.no_grad()
    def td_target(self, rewards, next_states, dones):
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        target_qs = self.net(next_states, model='target')
        target_qs = torch.max(target_qs, dim=1).values
        target_qs = target_qs.reshape(-1, 1)
        target_qs[dones] = 0.0
        return (rewards + (self.gamma * target_qs))

    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.target_network_sync_frequency == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.learning_starts:
            return None, None

        if self.curr_step % self.training_frequency != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"airstriker_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                replay_memory=self.memory
            ),
            save_path
        )

        print(f"Airstriker model saved to {save_path} at step {self.curr_step}")


    def load(self, load_path, reset_exploration_rate, load_replay_buffer):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)

        if load_replay_buffer:
            replay_memory = ckp.get('replay_memory')
            print(f"Loading replay memory. Len {len(replay_memory)}" if replay_memory else "Saved replay memory not found. Not restoring replay memory.")
            self.memory = replay_memory if replay_memory else self.memory

        if reset_exploration_rate:
            print(f"Reset exploration rate option specified. Not restoring saved exploration rate {exploration_rate}. The current exploration rate is {self.exploration_rate}")
        else:
            print(f"Setting exploration rate to {exploration_rate} not loaded.")
            self.exploration_rate = exploration_rate


class DDQNAgent(DQNAgent):
    @torch.no_grad()
    def td_target(self, rewards, next_states, dones):
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        q_vals = self.net(next_states, model='online')
        target_actions = torch.argmax(q_vals, axis=1)
        target_actions = target_actions.reshape(-1, 1)
        
        target_qs = self.net(next_states, model='target')
        target_qs = target_qs.gather(1, target_actions)
        target_qs = target_qs.reshape(-1, 1)
        target_qs[dones] = 0.0
        return (rewards + (self.gamma * target_qs))
