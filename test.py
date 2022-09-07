import argparse
import random
import time
from glob import glob

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.DnCNN import DnCNN
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test Nr2N public')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', default=10, type=int)

# Model parameters
parser.add_argument('--n_epochs', default=180, type=int)

# Test parameters
parser.add_argument('--noise', default='poisson_50', type=str)  # 'gauss_intensity', 'poisson_intensity'
parser.add_argument('--dataset', default='Set12', type=str)  # BSD100, Kodak, Set12
parser.add_argument('--aver_num', default=10, type=int)
parser.add_argument('--alpha', default=1.0, type=float)

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.4050)  # ImageNet Gray: 0.4050
parser.add_argument('--std', type=float, default=0.2927)  # ImageNet Gray: 0.2927

opt = parser.parse_args()


def generate(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model
    model = DnCNN().to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Directory
    img_dir = os.path.join('../all_datasets/', args.dataset)
    save_dir = os.path.join('./results/', args.dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Images
    img_paths = glob(os.path.join(img_dir, '*.png'))
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in img_paths]

    # Noise
    noise_type = args.noise.split('_')[0]
    noise_intensity = float(args.noise.split('_')[1]) / 255.

    # Transform
    transform = transforms.Compose(get_transforms(args))

    # Denoising
    noisy_psnr, output_psnr, prediction_psnr, overlap_psnr = 0, 0, 0, 0
    noisy_ssim, output_ssim, prediction_ssim, overlap_ssim = 0, 0, 0, 0

    avg_time1, avg_time2, avg_time3 = 0, 0, 0

    for index, clean255 in enumerate(imgs):
        if args.crop:
            clean255 = crop(clean255, patch_size=args.patch_size)

        clean_numpy = clean255/255.
        if noise_type == 'gauss':
            noisy_numpy = clean_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
            noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity * args.alpha
        elif noise_type == 'poisson':
            noisy_numpy = np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
            # 1. Add Poisson
            noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
            # 2. Add Gaussian approximation
            # noisier = noisy + np.random.randn(*clean.shape) *
        else:
            raise NotImplementedError('wrong type of noise')

        noisy, noisier = transform(noisy_numpy), transform(noisier_numpy)
        noisy, noisier = torch.unsqueeze(noisy, dim=0), torch.unsqueeze(noisier, dim=0)
        noisy, noisier = noisy.type(torch.FloatTensor).to(device), noisier.type(torch.FloatTensor).to(device)

        # Noisy Output
        start1 = time.time()
        # output = model(noisy)
        output = noisy
        elapsed1 = time.time() - start1
        avg_time1 += elapsed1 / len(imgs)

        # Noisier Prediction
        start2 = time.time()
        prediction = ((1 + args.alpha ** 2) * model(noisier) - noisier) / (args.alpha ** 2)
        # prediction = noisy
        elapsed2 = time.time() - start2
        avg_time2 += elapsed2 / len(imgs)

        # Overlap Prediction
        start3 = time.time()
        # # Method 1
        # overlap = None
        # for _ in range(args.aver_num):
        #     if noise_type == 'gauss':
        #         noisy_numpy = clean_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
        #         noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
        #     elif noise_type == 'poisson':
        #         noisy_numpy = np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
        #         # 1. Add Poisson
        #         noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
        #         # 2. Add Gaussian approximation
        #         # noisier = noisy + np.random.randn(*clean.shape) *
        #     else:
        #         raise NotImplementedError('wrong type of noise')
        #
        #     noisy, noisier = transform(noisy_numpy), transform(noisier_numpy)
        #     noisy, noisier = torch.unsqueeze(noisy, dim=0), torch.unsqueeze(noisier, dim=0)
        #     noisy, noisier = noisy.type(torch.FloatTensor).to(device), noisier.type(torch.FloatTensor).to(device)
        #
        #     with torch.no_grad():
        #         single_prediction = ((1 + args.alpha ** 2) * model(noisier) - noisier) / (args.alpha ** 2)
        #     if overlap is None:
        #         overlap = single_prediction.detach() / args.aver_num
        #     else:
        #         overlap += single_prediction.detach() / args.aver_num
        #     del single_prediction

        # overlap = noisy

        # Method 2
        noisier = torch.zeros(size=(args.aver_num, 1, *clean_numpy.shape))
        for i in range(args.aver_num):
            if noise_type == 'gauss':
                noisy_numpy = clean_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity
                noisier_numpy = noisy_numpy + np.random.randn(*clean_numpy.shape) * noise_intensity * args.alpha
            elif noise_type == 'poisson':
                noisy_numpy = np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255.
                noisier_numpy = noisy_numpy + (np.random.poisson(clean_numpy * 255. * noise_intensity) / noise_intensity / 255. - clean_numpy)
            else:
                raise NotImplementedError('wrong type of noise')

            noisier_tensor = transform(noisier_numpy)
            noisier_tensor = torch.unsqueeze(noisier_tensor, dim=0)
            noisier[i, :, :, :] = noisier_tensor

        noisier = noisier.type(torch.FloatTensor).to(device)
        overlap = ((1 + args.alpha ** 2)*model(noisier) - noisier) / (args.alpha ** 2)
        overlap = torch.mean(overlap, dim=0)

        elapsed3 = time.time() - start3
        avg_time3 += elapsed3 / len(imgs)

        # Change to Numpy
        if args.normalize:
            output = denorm(output, mean=args.mean, std=args.std)
            prediction = denorm(prediction, mean=args.mean, std=args.std)
            overlap = denorm(overlap, mean=args.mean, std=args.std)

        output, prediction, overlap = tensor_to_numpy(output), tensor_to_numpy(prediction), tensor_to_numpy(overlap)
        output_numpy, prediction_numpy, overlap_numpy = np.squeeze(output), np.squeeze(prediction), np.squeeze(overlap)

        # Calculate PSNR
        n_psnr = psnr(clean_numpy, noisy_numpy, data_range=1)
        o_psnr = psnr(clean_numpy, output_numpy, data_range=1)
        p_psnr = psnr(clean_numpy, prediction_numpy, data_range=1)
        op_psnr = psnr(clean_numpy, overlap_numpy, data_range=1)

        noisy_psnr += n_psnr / len(imgs)
        output_psnr += o_psnr / len(imgs)
        prediction_psnr += p_psnr / len(imgs)
        overlap_psnr += op_psnr / len(imgs)

        # Calculate SSIM
        n_ssim = ssim(clean_numpy, noisy_numpy, data_range=1)
        o_ssim = ssim(clean_numpy, output_numpy, data_range=1)
        p_ssim = ssim(clean_numpy, prediction_numpy, data_range=1)
        op_ssim = ssim(clean_numpy, overlap_numpy, data_range=1)

        noisy_ssim += n_ssim / len(imgs)
        output_ssim += o_ssim / len(imgs)
        prediction_ssim += p_ssim / len(imgs)
        overlap_ssim += op_ssim / len(imgs)

        print('{}th image | PSNR: noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f} | SSIM: noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f}'.format(
            index+1, n_psnr, o_psnr, p_psnr, op_psnr, n_ssim, o_ssim, p_ssim, op_ssim))

        # Save sample images
        if index <= 3:
            sample_clean, sample_noisy = 255. * np.clip(clean_numpy, 0., 1.), 255. * np.clip(noisy_numpy, 0., 1.)
            sample_output, sample_prediction = 255. * np.clip(output_numpy, 0., 1.), 255. * np.clip(prediction_numpy, 0., 1.)
            sample_overlap = 255. * np.clip(overlap_numpy, 0., 1.)
            cv2.imwrite(os.path.join(save_dir, '{}th_clean.png'.format(index+1)), sample_clean)
            cv2.imwrite(os.path.join(save_dir, '{}th_noisy.png'.format(index+1)), sample_noisy)
            cv2.imwrite(os.path.join(save_dir, '{}th_output.png'.format(index+1)), sample_output)
            cv2.imwrite(os.path.join(save_dir, '{}th_prediction.png'.format(index+1)), sample_prediction)
            cv2.imwrite(os.path.join(save_dir, '{}th_overlap.png'.format(index+1)), sample_overlap)

    # Total PSNR, SSIM
    print('{} Average PSNR | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f}'.format(
        args.dataset, noisy_psnr, output_psnr, prediction_psnr, overlap_psnr))
    print('{} Average SSIM | noisy:{:.3f}, output:{:.3f}, prediction:{:.3f}, overlap:{:.3f}'.format(
        args.dataset, noisy_ssim, output_ssim, prediction_ssim, overlap_ssim))
    print('Average Time for Output | denoised:{}'.format(avg_time1))
    print('Average Time for Prediction | denoised:{}'.format(avg_time2))
    print('Average Time for Overlap | denoised:{}'.format(avg_time3))


if __name__ == "__main__":
    generate(opt)
