from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class ImageNetGray(Dataset):
    def __init__(self, data_dir='../all_datasets/ImageNet_1000_Gray/', noise='gauss_25', train=True, transform=None):
        super(ImageNetGray, self).__init__()

        self.noise_type, self.noise_intensity = noise.split('_')[0], float(noise.split('_')[1]) / 255.

        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')

        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.
        if self.noise_type == 'gauss':
            noisy = clean + np.random.randn(*clean.shape) * self.noise_intensity
            noisier = noisy + np.random.randn(*clean.shape) * self.noise_intensity
        elif self.noise_type == 'poisson':
            noisy = np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255.
            # Add Poisson
            noisier = noisy + (np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255. - clean)
            # # Add Gaussian approximation
            # noisier = noisy + np.random.randn(*clean.shape) *
        else:
            raise NotImplementedError('wrong type of noise')
        clean, noisy, noisier = self.transform(clean), self.transform(noisy), self.transform(noisier)
        clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        return {'clean': clean, 'noisy': noisy, 'noisier': noisier}

    def __len__(self):
        return len(self.clean_paths)







