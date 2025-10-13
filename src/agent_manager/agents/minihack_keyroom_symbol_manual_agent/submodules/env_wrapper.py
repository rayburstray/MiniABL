from ...base_agent.submodules.env_wrapper import EnvWrapperInterface
import torch.nn as nn
import torch
from torchvision.transforms import ToTensor  # 导入ToTensor
from loguru import logger

model_path ='./src/agent_manager/agents/minihack_keyroom_symbol_manual_agent/submodules/cnn_model.pth'

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
        self.to_tensor = ToTensor()  # 初始化ToTensor变换
        self.device = device

    def forward(self, x):
        #logger.info(x.shape)
        x = torch.stack( [self.to_tensor(img) for img in x] )
        #x = self.to_tensor(x['pixel'])
        x = x.to(self.device)
        # 对x进行ToTensor的transform

        
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EnvWrapper(EnvWrapperInterface):
    def __init__(self, conf: dict):
        super().__init__(conf)
        self.model = SimpleCNN(num_classes=7, device=self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)


    def process(self, obs, reward, terminated, truncated, info):
        parallel = self.conf['envs']['parallel']
        # 存储obs
        for i in range(parallel):
            obs_list = self.memory.get_memory(i, 'obs_list')
            obs_list.append(obs)
            self.memory.update(i, 'obs_list', obs_list)

        grid_size = (16, 16)
        w, h = obs['pixel'].shape[1]//grid_size[0], obs['pixel'].shape[2]//grid_size[1]
        obs_processed = torch.zeros(obs['pixel'].shape[0], w, h)
        for i in range(w):
            for j in range(h):
                obs_processed[:, i, j] = torch.argmax(self.model(obs['pixel'][:, i*grid_size[0]:(i+1)*grid_size[0], j*grid_size[1]:(j+1)*grid_size[1]]), dim=1)
        
        return obs_processed, reward, terminated, truncated, info