from hmac import new
from cv2 import log
from ...base_agent.submodules.action_model import ActionModelInterface
from loguru import logger
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.transforms import ToTensor  # 导入ToTensor
import torch
import yaml
import os

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module): 
    """
    适配Minigrid任务的PPO网络（Actor-Critic结构）
    输入：(batch_size, 64, 64, 3) → 输出：6个离散动作的概率分布 + 状态价值
    """
    def __init__(self, conf: dict):
        super().__init__()
        # 注意，这里传入的conf是最外层的那个conf，这里我们其实需要的是这个ppo-agent专属的conf文件
        new_conf_path = 'src/agent_manager/agents/minigrid_keyroom_ppo_agent/conf.yaml'
        new_conf = yaml.load(open(new_conf_path), Loader=yaml.FullLoader)
        full_conf = {**conf, **new_conf}
        self.conf = full_conf
        self.input_shape = self.conf['action_model']['input_shape']
        self.output_shape = self.conf['action_model']['output_shape']
        self.device = self.conf['device']

        self.totensor = ToTensor()
        
        self.shared_cnn = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算CNN输出特征维度（用于后续全连接层）
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 64, 64)  # (batch, channel, height, width)
            cnn_output = self.shared_cnn(dummy_input)
            self.cnn_feature_dim = cnn_output.shape[1]

        #  Critic网络（价值函数）
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cnn_feature_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),  # 输出状态价值
        )
        
        # Actor网络（策略函数）
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.cnn_feature_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, self.output_shape), std=0.01),  # 输出动作 logits
        )

    def get_value(self, x):
        x = torch.stack( [self.totensor(obs) for obs in x] )
        x = x.to(self.device)
        x = self.shared_cnn(x)
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        x = torch.stack( [self.totensor(obs) for obs in x] )
        x = x.to(self.device)
        x = self.shared_cnn(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

        pass


    

class ActionModel(ActionModelInterface):
    def __init__(self, conf):
        super().__init__(conf)
        self.model = PPO(conf)
        self.conf = self.model.conf
        self.parallel = self.conf['envs']['parallel']
        self.device = torch.device(conf['device'])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf['action_model']['lr'], eps=1e-5)

        ppo_path = self.conf['action_model']['save_path']
        if os.path.exists(ppo_path):
            logger.info('已为您加载上次的PPO权重')
            self.model.load_state_dict(torch.load(ppo_path))
        


    def act(self, obs_processed):
        '''
        仅涉及到inference， 不需要考虑train
        '''
        actions, _, _, _ = self.model.get_action_and_value(obs_processed)
        return actions.cpu().numpy()
        
    def learn(self, obs_processed, done_processed, env):
        
        if self.conf['action_model']['anneal_lr']:
            iteration = self.memory.get_memory(0, 'iteration')
            max_steps = self.memory.get_memory(0, 'max_steps')
            frac = 1.0 - (iteration - 1.0) / max_steps
            lrnow = frac * self.conf['action_model']['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrnow

            for i in range(self.parallel):
                history_obs = self.memory.get_memory(i, 'obs')
                history_obs.append(obs_processed[i])
                self.memory.update(i, 'obs', history_obs)

                if done_processed is not None:
                    history_done = self.memory.get_memory(i, 'done')
                    history_done.append(done_processed[i])
                    self.memory.update(i, 'done', history_done)
                else:
                    history_done = self.memory.get_memory(i, 'done')
                    history_done.append(0)
                    self.memory.update(i, 'done', history_done)


        with torch.no_grad():
            action, logprob, _, value = self.model.get_action_and_value(obs_processed)
            for i in range(self.parallel):
                history_action = self.memory.get_memory(i, 'action')
                history_action.append(action[i].cpu().numpy())
                self.memory.update(i, 'action', history_action)

                history_logprob = self.memory.get_memory(i, 'logprob')
                history_logprob.append(logprob[i].cpu().numpy())
                self.memory.update(i, 'logprob', history_logprob)

                history_value = self.memory.get_memory(i, 'value')
                #logger.info(value.reshape(-1).shape)
                history_value.append(value.reshape(-1)[i].cpu().numpy())
                self.memory.update(i, 'value', history_value)
        
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()) 
        if np.any(reward > 0):
            pass
            #logger.info('卧槽666')
        next_obs = next_obs['image']
        for i in range(self.parallel):
            history_reward = self.memory.get_memory(i, 'reward')
            history_reward.append(reward[i])    
            self.memory.update(i, 'reward', history_reward)

            
        next_done = np.logical_or(terminated, truncated).astype(int)
        #logger.info(next_done)
        

        return next_obs, next_done, env


    


    def update(self, next_obs, next_done):

        with torch.no_grad():
            next_value = self.model.get_value(next_obs).reshape(-1)
            obss = np.array([self.memory.get_memory(i, 'obs') for i in range(self.parallel)]).transpose(1, 0, 2, 3, 4)
            logprobs = np.array([self.memory.get_memory(i, 'logprob') for i in range(self.parallel)]).transpose(1, 0)
            rewards = np.array([self.memory.get_memory(i, 'reward') for i in range(self.parallel)]).transpose(1, 0)
            values = np.array([self.memory.get_memory(i, 'value') for i in range(self.parallel)]).transpose(1, 0)
            dones = np.array([self.memory.get_memory(i, 'done') for i in range(self.parallel)]).transpose(1, 0)
            actions = np.array([self.memory.get_memory(i, 'action') for i in range(self.parallel)]).transpose(1, 0)
            
            #obss = torch.tensor(obss).to(self.device)
            logprobs = torch.tensor(logprobs).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            values = torch.tensor(values).to(self.device)
            dones = torch.tensor(dones).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            #logger.info(dones)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(self.conf['envs']['steps_per_train'])):
                if t == self.conf['envs']['steps_per_train'] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = [self.memory.get_memory(i, 'value')[t + 1].item() for i in range(self.parallel)]
                    #logger.info(nextvalues)
                    nextvalues = torch.tensor(nextvalues).to(self.device)
                nextnonterminal = torch.tensor(nextnonterminal).to(self.device)

                delta = rewards[t] + self.conf['action_model']['gamma'] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.conf['action_model']['gamma'] * self.conf['action_model']['gae_lambda'] * nextnonterminal * lastgaelam
            returns = advantages + values
        
        b_obs = obss.reshape((-1, ) + tuple(self.model.input_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        batch_size = int( self.parallel * self.conf['envs']['steps_per_train'] )
        minibatch_size = int( batch_size // self.conf['action_model']['num_minibatchs'] )
        num_iterations = self.conf['action_model']['total_timesteps'] // batch_size

        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range( self.conf['action_model']['update_epochs'] ):
            np.random.shuffle(b_inds)
            for start in range( 0, batch_size, minibatch_size ):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.conf['action_model']['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.conf['action_model']['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.conf['action_model']['clip_coef'], 1 + self.conf['action_model']['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.conf['action_model']['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.conf['action_model']['clip_coef'],
                        self.conf['action_model']['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.conf['action_model']['ent_coef'] * entropy_loss + v_loss * self.conf['action_model']['vf_coef']

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf['action_model']['max_grad_norm'])
                self.optimizer.step()

            if self.conf['action_model']['target_kl'] is not None and approx_kl > self.conf['action_model']['target_kl']:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        #logger.info('训练模型结束一轮')
        torch.save(self.model.state_dict(), self.conf['action_model']['save_path'])
        
        for i in range(self.parallel):
            self.memory.update(i, 'obs', [])
            self.memory.update(i, 'done', [])
            self.memory.update(i, 'reward', [])
            self.memory.update(i, 'action', [])
            self.memory.update(i, 'logprob', [])
            self.memory.update(i, 'value', [])
        pass