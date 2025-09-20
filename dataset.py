from utils import *
import matplotlib.pyplot as plt
import os
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TrainSetLoader(Dataset):
    """Unified training dataset loader with configurable augmentations.

    Args:
        dataset_dir: Root directory of datasets
        dataset_name: Name of the specific dataset
        patch_size: Size of patches to extract
        img_norm_cfg: Image normalization configuration (optional)
        add_noise: Whether to add random noise augmentation (default: False)
        add_gamma: Whether to add gamma correction augmentation (default: False)
        noise_std: Standard deviation of noise to add (default: 0.03)
        gamma_range: Range for gamma correction (default: (0.5, 1.6))
    """

    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None,
                 add_noise=False, add_gamma=False, noise_std=0.03, gamma_range=(0.5, 1.6)):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        self.add_noise = add_noise
        self.add_gamma = add_gamma
        self.noise_std = noise_std
        self.gamma_range = gamma_range

        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

        self.transform = Augmentation()

    def __getitem__(self, idx):
        try:
            # 构建基础文件名
            base_name = self.train_list[idx]
            img_pattern = f"{self.dataset_dir}/images/{'._' if '._' in base_name else ''}{base_name}.png"
            mask_pattern = f"{self.dataset_dir}/masks/{'._' if '._' in base_name else ''}{base_name}*pixels0.png"
            
            # 尝试查找匹配的掩码文件
            mask_files = glob.glob(mask_pattern.replace('//', '/'))
            if mask_files:
                mask_path = mask_files[0]
            else:
                mask_path = f"{self.dataset_dir}/masks/{base_name}.png".replace('//', '/')
            
            img = Image.open(img_pattern.replace('//', '/')).convert('I')
            mask = Image.open(mask_path)
        except:
            try:
                # 尝试 bmp 格式
                img_pattern = f"{self.dataset_dir}/images/{'._' if '._' in base_name else ''}{base_name}.bmp"
                mask_pattern = f"{self.dataset_dir}/masks/{'._' if '._' in base_name else ''}{base_name}*pixels0.bmp"
                
                mask_files = glob.glob(mask_pattern.replace('//', '/'))
                if mask_files:
                    mask_path = mask_files[0]
                else:
                    mask_path = f"{self.dataset_dir}/masks/{base_name}.bmp".replace('//', '/')
                
                img = Image.open(img_pattern.replace('//', '/')).convert('I')
                mask = Image.open(mask_path)
            except Exception as e:
                print(f"Error loading file {base_name}: {str(e)}")
                raise e

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # Apply random noise if enabled
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std)
            img += noise

        # Apply gamma correction if enabled
        if self.add_gamma:
            min_val = img.min()
            max_val = img.max()
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            img = np.power(((img - min_val) / (max_val - min_val)), gamma)
            img = img * (max_val - min_val) + min_val

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5)  # Pad the shorter side first, then randomly crop to get 256x256 output

        img_patch, mask_patch = self.transform(img_patch, mask_patch)  # Data augmentation with flipping
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]  # 升维
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))  # numpy 转tensor
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))  # numpy 转tensor
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)

# TrainSetLoader02 functionality is now integrated into main TrainSetLoader class
# Use TrainSetLoader(dataset_dir, dataset_name, patch_size, add_noise=True) for noise augmentation


# TrainSetLoader03 functionality is now integrated into main TrainSetLoader class
# Use TrainSetLoader(dataset_dir, dataset_name, patch_size, add_gamma=True) for gamma correction augmentation


# TrainSetLoader04 functionality is now integrated into main TrainSetLoader class
# Use TrainSetLoader(dataset_dir, dataset_name, patch_size, add_noise=True, add_gamma=True) for both augmentations

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir 
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
        # with open(r'D:\05TGARS\Upload\datasets\SIRST3\img_idx\val.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(test_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        try:
            base_name = self.test_list[idx]
            img_pattern = f"{self.dataset_dir}/images/{'._' if '._' in base_name else ''}{base_name}.png"
            mask_pattern = f"{self.dataset_dir}/masks/{'._' if '._' in base_name else ''}{base_name}*pixels0.png"
            
            mask_files = glob.glob(mask_pattern.replace('//', '/'))
            if mask_files:
                mask_path = mask_files[0]
            else:
                mask_path = f"{self.dataset_dir}/masks/{base_name}.png".replace('//', '/')
            
            img = Image.open(img_pattern.replace('//', '/')).convert('I')
            mask = Image.open(mask_path)
        except:
            try:
                img_pattern = f"{self.dataset_dir}/images/{'._' if '._' in base_name else ''}{base_name}.bmp"
                mask_pattern = f"{self.dataset_dir}/masks/{'._' if '._' in base_name else ''}{base_name}*pixels0.bmp"
                
                mask_files = glob.glob(mask_pattern.replace('//', '/'))
                if mask_files:
                    mask_path = mask_files[0]
                else:
                    mask_path = f"{self.dataset_dir}/masks/{base_name}.bmp".replace('//', '/')
                
                img = Image.open(img_pattern.replace('//', '/')).convert('I')
                mask = Image.open(mask_path)
            except Exception as e:
                print(f"Error loading file {base_name}: {str(e)}")
                raise e

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # if mask.shape == (416,608):
        #     print('111')
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if img.size() != mask.size():
            print('111')
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open(
            (self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//', '/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0

        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:, :, 0]

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]

    def __len__(self):
        return len(self.test_list)


class Augmentation(object):
    """Data augmentation class for random transformations."""

    def __call__(self, input_img, target_img):
        """Apply random augmentations to input and target images.

        Args:
            input_img: Input image array
            target_img: Target/label image array

        Returns:
            Tuple of (augmented_input, augmented_target)
        """
        # Horizontal flip (left-right)
        if random.random() < 0.5:
            input_img = input_img[::-1, :]
            target_img = target_img[::-1, :]

        # Vertical flip (up-down)
        if random.random() < 0.5:
            input_img = input_img[:, ::-1]
            target_img = target_img[:, ::-1]

        # Transpose (swap axes)
        if random.random() < 0.5:
            input_img = input_img.transpose(1, 0)
            target_img = target_img.transpose(1, 0)

        return input_img, target_img
