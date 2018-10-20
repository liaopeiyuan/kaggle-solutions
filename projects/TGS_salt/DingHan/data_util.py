print("importng common in data.py")
from common import *

DATA_DIR = "E:\\DHWorkStation\\Project\\tgs_pytorch\\data"
IMAGE_HEIGHT, IMAGE_WIDTH = 101, 101
HEIGHT, WIDTH = 128, 128
#DATA_DIR = '/root/share/project/kaggle/tgs/data'
def null_augment(image, label, index):
    cache = Struct(image=image.copy(), mask=mask.copy())
    return image, label, index, cache


def null_collate(batch):
    batch_size = len(batch)
    cache = []
    input = []
    truth = []
    index = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        index.append(batch[b][2])
        cache.append(batch[b][3])
    input = torch.from_numpy(np.array(input)).float().unsqueeze(1)

    if truth[0] != []:
        truth = torch.from_numpy(np.array(truth)).float().unsqueeze(1)

    return input, truth, index, cache


# ----------------------------------------
def valid_augment(image, mask, index):
    cache = Struct(image=image.copy(), mask=mask.copy())
    image, mask = do_center_pad_to_factor2(image, mask, factor=32)
    return image, mask, index, cache


# ----------------------------------------
class TsgDataset(Dataset):
    print('class')
    def __init__(self, split, augment=null_augment, mode='train'):
        print('init')
        super(TsgDataset, self).__init__()
        print('super')
        self.split = split
        self.mode = mode
        self.augment = augment

        split_file = DATA_DIR + '/split/' + split
        lines = read_list_from_file(split_file)

        self.ids = []
        self.images = []
        print('arg_init')
        for l in lines:
            folder, name = l.split('/')
            image_file = DATA_DIR + '/' + folder + '/images/' + name + '.png'
            try:
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            except:
                print(image_file)
                print("wrong")
                exit(-1)

            self.images.append(image)
            self.ids.append(name)
            # print(image.shape)

        self.masks = []
        if self.mode in ['train', 'valid']:
            for l in lines:
                folder, file = l.split('/')
                mask_file = DATA_DIR + '/' + folder + '/masks/' + file + '.png'
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                self.masks.append(mask)
        elif self.mode in ['test']:
            self.masks = [[] for l in lines]

        # -------
        df = pd.read_csv(DATA_DIR + '/depths.csv')
        df = df.set_index('id')
        self.zs = df.loc[self.ids].z.values

        # -------
        print('\tTsgDataset')
        print('\tsplit            = %s' % split)
        print('\tlen(self.images) = %d' % len(self.images))
        print('')

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        return self.augment(image, mask, index)

    def __len__(self):
        return len(self.images)

def run_check_data0():
    # dataset = TsgDataset('list_train_4000', mode='train')
    dataset = TsgDataset('list_test_18000', mode='test')

    # --
    zz = 0

    zero = np.zeros((101, 101), np.uint8)

    save_dir = '/root/share/project/kaggle/tgs/results/xxx'
    num = len(dataset)
    for m in range(num):
        if dataset.zs[m] > 150: continue

        image = (dataset.images[m] * 255).astype(np.uint8)
        mask = zero  # (dataset.masks [m]*255).astype(np.uint8)

        # https://github.com/flrs/blend_modes
        alpha = 0.5
        all = np.dstack([
            image,
            image * (1 - mask / 255),
            image,
            # alpha*image+(1-alpha)*mask,
        ])

        cv2.imwrite('/root/share/project/kaggle/tgs/results/xxx/%05d_%s_.png' % (
            dataset.zs[m], dataset.ids[m])
                    , all)

def run_check_data():
    dataset = TsgDataset('list_train_4000', mode='train')  #

    # --
    zz = 0
    zero = np.zeros((101, 101), np.uint8)
    save_dir = '/root/share/project/kaggle/tgs/results/xxx'
    num = len(dataset)
    for m in [3, 5, 6, 7, 8, 9, 10, 11, 12]:

        image = dataset.images[m]
        mask = dataset.masks[m]
        image_show_norm('image', image, 1, 2)
        image_show_norm('mask', mask, 1, 2)

        for i in range(5):
            # image1, mask1 = do_random_pad_to_factor2(image, mask, limit=(-4,4), factor=32)
            # image1, mask1 = do_horizontal_flip2(image, mask)

            mask1 = mask
            # image1 = do_invert_intensity(image)
            # image1 = do_brightness_shift(image, np.random.uniform(-0.125,0.125))
            # image1 = do_brightness_multiply(image, np.random.uniform(1-0.125,1+0.125))
            image1 = do_gamma(image, np.random.uniform(1 - 0.25, 1 + 0.25))

            # -----------------------------------------------
            image1 = (image1 * 255).astype(np.uint8)
            image1 = np.dstack([image1, image1, image1])
            overlay1 = draw_mask_overlay(mask1, image1, color=[0, 0, 255])
            image_show('overlay1', overlay1, 2)
            image_show('image1', image1, 2)
            image_show_norm('mask1', mask1, 1, 2)
            cv2.waitKey(0)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_data()

