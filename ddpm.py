## reverse에서 각 타입스텝마다 noise를 예측하고 이를 Xt에서 뺴서 Xt-1을 생성 --> 반복

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import *
from modules import UNet



logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


## x가 인풋되면 여기에 sample_timestep으로 뽑힌 n개의 에러 noising을 x에 진행하고 --> 여기서 다시 x로 복원하는 과정을 학습
## 

class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #cumulative production on 1-beta

    def prepare_noise_schedule(self):
        # linspace는 beta_start~beta_end까지 noise_steps만큼 even하게 만들어줌 --> linear하게 beta줄도록 scheduling
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        # tstep 만큼 x에 noise를 가할건데, 한번에 보내는 수식을 쓸거임
        # alpha_hat^(1/2) * x0 + (1-alpha_hat)^(1/2) * rand_noise

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        #여기서 None을 추가하는 이유는 batch가 1이든 그거보다 크든, 항상 스칼라처럼 곱하기 위함임
        #파이토치나 넘파이는 broad_casting으로 None으로 연장된 차원의 벡터를 전체에 스칼라처럼 곱하는 broadcasting지원

        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x) #x의 차원과동일한 공간을 rand num ~ [0,1) 으로 채움

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    ## DDPM논문의 ALgorithm2 부분
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images ...")
        model.eval()
        
        #eval만하면 gradient를 여전히 자동계산하기 때문에, 메모리를 효율적으로 쓰기위해 no_grad도 한번더 해줌
        with torch.no_grad():
            #initial image생성 = gausian noise(X_T)
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)    
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                #현재 timestep i의 vector를 long type으로 생성 = t, n은 샘플수
                t = (torch.ones(n) * i).long.to(self.device)
                #해당 timestep의 noise를 예측
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    #생성후 더해주는 noise인데, 마지막 step 즉 i==0에서는 원하는 이미지가 생성될거니까 noise==0으로 진행
                    noise = torch.zeros_like(x)
                #
                x = 1/torch.sqrt(alpha) * (x - ((1-alpha)/(torch.sqrt(1-alpha_hat)))* predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        #-1,1로 clip한 후에 1더하고 2로나눠서 0,1로 범위바꿈
        x = (x.clamp(-1, 1) + 1)/2
        x = (x*255).type(torch.uint8)

        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    ##여기부터 DDPM의 algorithm1 부분
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0].to(device))
            #noise도 아웃풋하는 이유는 loss계산을 위해서이고, t를 통해 해당 스텝의 noising을 이미지에 해줌
            x_t, noise = diffusion.noise_images(images, t)
            #이걸로 한번에 input image를 t-step의 noisy한 이미지로 매핑함(forward)

            #이후 이 noisy 이미지를 model에 넣어서 noise를 예측함
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*l + i)
        
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    #여기서 parser.add_argument('--input', type=str, help='Input file path') 같이 해서 명령줄의 입력을 args에 담을 수 있음
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r""
    args.device = "cuda:0"
    args.lr = 3e-4
    train(args)


if __name__ == "__main__":
    launch()


