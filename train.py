import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import json
import random
from tqdm import tqdm

from utils import *
from models.DnCNN import DnCNN
from dataset import ImageNetGray


class TrainNr2N:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.start_epoch = args.start_epoch
        self.decay_epoch = args.decay_epoch
        self.lr = args.lr
        self.noise = args.noise
        self.noise_type = self.noise.split('_')[0]
        self.noise_intensity = float(self.noise.split('_')[1]) / 255.

        # Loss
        self.criterion_mse = nn.MSELoss()

        # Transformation Parameters
        self.mean = args.mean
        self.std = args.std

        # Transform
        transform = transforms.Compose(get_transforms(args))

        # Models
        self.model = DnCNN().to(self.device)
        if args.load_model:
            load_path = './experiments/exp{}/checkpoints/{}epochs.pth'.format(args.load_exp_num, args.load_epoch)
            self.model.load_state_dict(torch.load(load_path))

        # Dataset
        self.train_dataset = ImageNetGray(noise=self.noise, train=True, transform=transform)
        self.test_dataset = ImageNetGray(noise=self.noise, train=False, transform=transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.n_epochs, self.start_epoch, self.decay_epoch).step)

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                for batch, data in enumerate(tepoch):
                    self.model.train()
                    self.optimizer.zero_grad()

                    clean, noisy, noisier = data['clean'], data['noisy'], data['noisier']
                    clean, noisy, noisier = clean.to(self.device), noisy.to(self.device), noisier.to(self.device)

                    prediction = self.model(noisier)
                    loss = self.criterion_mse(prediction, noisy)
                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(rec_loss=loss.item())
                    self.summary.add_scalar('loss', loss.item(), epoch)

            self.scheduler.step()

            # Checkpoints
            if epoch % 10 == 0 or epoch == self.n_epochs:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

            if epoch % 5 == 0:
                noisy_psnr, output_psnr, prediction_psnr = 0, 0, 0
                noisy_ssim, output_ssim, prediction_ssim = 0, 0, 0

                with torch.no_grad():
                    self.model.eval()

                    num_data = 10
                    for index in range(num_data):
                        data = self.test_dataset[index]
                        sample_clean, sample_noisy, sample_noisier = data['clean'], data['noisy'], data['noisier']
                        sample_noisy = torch.unsqueeze(sample_noisy, dim=0).to(self.device)
                        sample_noisier = torch.unsqueeze(sample_noisier, dim=0).to(self.device)

                        sample_output = self.model(sample_noisy)
                        sample_prediction = 2*self.model(sample_noisier) - sample_noisier

                        if self.args.normalize:
                            sample_clean = denorm(sample_clean, mean=self.mean, std=self.std)
                            sample_noisy = denorm(sample_noisy, mean=self.mean, std=self.std)
                            sample_output = denorm(sample_output, mean=self.mean, std=self.std)
                            sample_prediction = denorm(sample_prediction, mean=self.mean, std=self.std)

                        sample_clean, sample_noisy = tensor_to_numpy(sample_clean), tensor_to_numpy(sample_noisy)
                        sample_output, sample_prediction = tensor_to_numpy(sample_output), tensor_to_numpy(sample_prediction)

                        sample_clean, sample_noisy = np.squeeze(sample_clean), np.squeeze(sample_noisy)
                        sample_output, sample_prediction = np.squeeze(sample_output), np.squeeze(sample_prediction)

                        # Calculate PSNR
                        n_psnr = psnr(sample_clean, sample_noisy, data_range=1)
                        o_psnr = psnr(sample_clean, sample_output, data_range=1)
                        p_psnr = psnr(sample_clean, sample_prediction, data_range=1)
                        # print('{}th image PSNR | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(index + 1, n_psnr, o_psnr, p_psnr))

                        noisy_psnr += n_psnr / num_data
                        output_psnr += o_psnr / num_data
                        prediction_psnr += p_psnr / num_data

                        # Calculate SSIM
                        n_ssim = ssim(sample_clean, sample_noisy, data_range=1)
                        o_ssim = ssim(sample_clean, sample_output, data_range=1)
                        p_ssim = ssim(sample_clean, sample_prediction, data_range=1)
                        # print('{}th image SSIM | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(index + 1, n_ssim, o_ssim, p_ssim))

                        noisy_ssim += n_ssim / num_data
                        output_ssim += o_ssim / num_data
                        prediction_ssim += p_ssim / num_data

                        # Save sample image
                        sample_clean, sample_noisy = 255. * np.clip(sample_clean, 0., 1.), 255. * np.clip(sample_noisy, 0., 1.)
                        sample_output, sample_prediction = 255. * np.clip(sample_output, 0., 1.), 255. * np.clip(sample_prediction, 0., 1.)

                        if index == 0:
                            cv2.imwrite(os.path.join(self.result_path, 'clean_{}epochs.png'.format(epoch)), sample_clean)
                            cv2.imwrite(os.path.join(self.result_path, 'noisy_{}epochs.png'.format(epoch)), sample_noisy)
                            cv2.imwrite(os.path.join(self.result_path, 'output_{}epochs.png'.format(epoch)), sample_output)
                            cv2.imwrite(os.path.join(self.result_path, 'prediction_{}epochs.png'.format(epoch)), sample_prediction)

                    # PSNR, SSIM
                    print('Average PSNR | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(noisy_psnr, output_psnr, prediction_psnr))
                    print('Average SSIM | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}'.format(noisy_ssim, output_ssim, prediction_ssim))
                    self.summary.add_scalar('avg_output_psnr', output_psnr, epoch)
                    self.summary.add_scalar('avg_output_ssim', output_ssim, epoch)
                    self.summary.add_scalar('avg_prediction_psnr', prediction_psnr, epoch)
                    self.summary.add_scalar('avg_prediction_ssim', prediction_ssim, epoch)

        self.summary.close()










