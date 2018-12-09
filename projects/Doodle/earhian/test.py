import datetime
from timeit import default_timer as timer
from dataSet.reader import *
from dataSet.transform import *
from models.model import *
import torch
import torch.nn as nn
import time
from utils.file import *
from utils.metric import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
size = 96
d, WIDTH, HEIGHT =3, 112, 112
num_TTA = 2 * 5
def train_collate(batch):
    batch_size = len(batch)
    images = []
    names = []
    for index in range(batch_size):
        images.extend(batch[index][0])
        names.append(batch[index][1])
    images = torch.stack(images, 0)
    return [images, names]
def transform_test(drawing):
    ori_image = drawing_to_image_with_color_v2(drawing, H=size, W=size)
    images = TTA_cropps(ori_image, target_shape=(WIDTH, HEIGHT, d))
    return images
checkPoint_start = 192000
resultDir = './result/xception'
checkPoint = os.path.join(resultDir, 'checkpoint')
model = model_QDDR(model_name='xception').cuda()
batch_size = 64 * 2

if not checkPoint_start == 0:
    model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=[])

dst_test = QDDRDataset(mode='test', transform=transform_test, size=112)
dataloader_test = DataLoader(dst_test, num_workers=8, batch_size=batch_size, collate_fn=train_collate)

labels = {value:key for key, value in dst_test.dict_label.items()}
testnames = []
rs = []
with torch.no_grad():
    model.eval()
    for data in tqdm(dataloader_test):
        images, names = data
        images = images.cuda()
        results_TTA = model(images)
        results = 0
        for index in range(num_TTA):
            results += results_TTA[index::num_TTA,]
        results_k_index = get_top_k(results, topk=(5, )).data.cpu().numpy()
        results_softmax = torch.softmax(results, 1).data.cpu().numpy()
        for res_k_index, result_softmax, name in zip(results_k_index, results_softmax, names):
            top_k_label_name = ''
            for r in res_k_index:
                label = str(labels[int(r)])
                score = str(result_softmax[int(r)])
                label = label.replace(' ', '_')
                top_k_label_name += label + ' ' + score + ' '
            testnames.append(name)
            rs.append(top_k_label_name)

    pd.DataFrame({'key_id':testnames, 'word':rs}).to_csv(
        os.path.join(resultDir, 'result_{}.csv'.format(checkPoint_start)), index=None
    )
