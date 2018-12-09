from common import *
from imgaug import augmenters as iaa


CLASS_NAME=\
['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'airplane', 'alarm_clock', 'ambulance', 'angel',
 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn',
 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee',
 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel',
 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup',
 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship',
 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser',
 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses',
 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo',
 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan',
 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger',
 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant',
 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop',
 'leaf', 'leg', 'light_bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox',
 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito',
 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants',
 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond',
 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'river', 'roller_coaster', 'rollerskates', 'sailboat',
 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw',
 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear',
 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle',
 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch',
 'yoga', 'zebra', 'zigzag']


#small dataset for debug
#CLASS_NAME = ['apple','bee', 'cat', 'fish', 'frog', 'leaf']

NUM_CLASS = len(CLASS_NAME)
TRAIN_DF  = []
TEST_DF   = []




def null_augment(drawing,label,index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 64, 64)
    return image, label, cache


def null_collate(batch):
    batch_size = len(batch)
    cache = []
    input = []
    truth = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        cache.append(batch[b][2])

    input = np.array(input).transpose(0,3,1,2)
    input = torch.from_numpy(input).float()

    if truth[0] is not None:
        truth = np.array(truth)
        truth = torch.from_numpy(truth).long()

    return input, truth, cache


#----------------------------------------

def drawing_to_image(drawing, H, W):

    point=[]
    time =[]
    for t,(x,y) in enumerate(drawing):
        point.append(np.array((x,y),np.float32).T)
        time.append(np.full(len(x),t))

    point = np.concatenate(point).astype(np.float32)
    time  = np.concatenate(time ).astype(np.int32)

    #--------
    image  = np.full((H,W,3),0,np.uint8)
    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min
    #print(w,h)

    s = max(w,h)
    norm_point = (point-[x_min,y_min])/s
    norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)


    #--------
    T = time.max()+1
    for t in range(T):
        p = norm_point[time==t]
        x,y = p.T
        image[y,x]=255
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image,(x0,y0),(x1,y1),(255,255,255),1,cv2.LINE_AA)

    return image

#each channel, has its own grey scale, can input any channel number, like 2,3 8
def drawing_to_image_with_color_aug_multi_channel_grey(drawing, H, W, seq, channels):
    cts = int(channels)

    point=[]
    time =[]
    for t,(x,y) in enumerate(drawing):
        point.append(np.array((x,y),np.float32).T)
        time.append(np.full(len(x),t))

    point = np.concatenate(point).astype(np.float32)
    time  = np.concatenate(time ).astype(np.int32)
    T = time.max()+1
    image_all = np.full((H,W,1),0,np.uint8)

    for ct in range(cts):
        image_tmp = np.full((H,W,1),0,np.uint8)
        x_max = point[:,0].max()
        x_min = point[:,0].min()
        y_max = point[:,1].max()
        y_min = point[:,1].min()
        w = x_max-x_min
        h = y_max-y_min
        #print(w,h)

        s = max(w,h)
        norm_point = (point-[x_min,y_min])/s
        norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
        norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)

        P_part = 0
        for t in range(int(ct*T/cts), int((ct+1)*T/cts)):
            P_part += len(norm_point[time==t])

        if P_part > 0:
            #use grey color
            colors = plt.cm.gray(np.arange(0,P_part+1)/(P_part))
            colors = (colors[:,0]*255).astype(np.uint8)

        p_num = 0
        for t in range(int(ct*T/cts), int((ct+1)*T/cts)):
            p = norm_point[time==t]
            x,y = p.T
            image_tmp[y,x]=255
            N = len(p)
            for i in range(N-1):
                #only 1-D color is used for one channel
                color = colors[p_num]
                x0,y0 = p[i]
                x1,y1 = p[i+1]
                cv2.line(image_tmp,(x0,y0),(x1,y1),int(color),1,cv2.LINE_AA)
                p_num += 1

        if ct == 0:
            image_all = image_tmp
        else:
            image_all = np.concatenate((image_all,image_tmp),axis=2)

    image_all = seq.augment_image(image_all)
    return image_all


#each 3 channels, has their RGB colors, can input 3,6,9...
def drawing_to_image_with_color_aug_multi_channel(drawing, H, W, seq, channels):
	cts = int(channels/3)

	point=[]
    time =[]
    for t,(x,y) in enumerate(drawing):
        point.append(np.array((x,y),np.float32).T)
        time.append(np.full(len(x),t))

    point = np.concatenate(point).astype(np.float32)
    time  = np.concatenate(time ).astype(np.int32)
    T = time.max()+1
    image_all = np.full((H,W,3),0,np.uint8)

    for ct in range(cts):
    	image_tmp = np.full((H,W,3),0,np.uint8)
	    x_max = point[:,0].max()
	    x_min = point[:,0].min()
	    y_max = point[:,1].max()
	    y_min = point[:,1].min()
	    w = x_max-x_min
	    h = y_max-y_min
	    #print(w,h)

	    s = max(w,h)
	    norm_point = (point-[x_min,y_min])/s
	    norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
	    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)
	    
	    #produce colors
	    P_part = 0
	    for t in range(int(ct*T/cts), int((ct+1)*T/cts)):
	        P_part += len(norm_point[time==t])
	    if P_part > 0:
		    colors = plt.cm.jet(np.arange(0,P_part+1)/(P_part))
		    colors = (colors[:,:3]*255).astype(np.uint8)
		#draw
	    p_num = 0
	    for t in range(int(ct*T/cts), int((ct+1)*T/cts)):
	        p = norm_point[time==t]
	        x,y = p.T
	        image_tmp[y,x]=255
	        N = len(p)
	        for i in range(N-1):
	            color = colors[p_num]
	            color = [int(color[2]),int(color[1]),int(color[0])]
	            x0,y0 = p[i]
	            x1,y1 = p[i+1]
	            cv2.line(image_tmp,(x0,y0),(x1,y1),color,1,cv2.LINE_AA)
	            p_num += 1

	    if ct == 0:
	    	image_all = image_tmp
	    else:
	    	image_all = np.concatenate((image_all,image_tmp),axis=2)

	image_all = seq.augment_image(image_all)
	return image_all


def drawing_to_image_with_color_aug_6channel(drawing, H, W, seq):

    point=[]
    time =[]
    for t,(x,y) in enumerate(drawing):
        point.append(np.array((x,y),np.float32).T)
        time.append(np.full(len(x),t))

    point = np.concatenate(point).astype(np.float32)
    time  = np.concatenate(time ).astype(np.int32)

    #--------
    image1  = np.full((H,W,3),0,np.uint8)
    image2  = np.full((H,W,3),0,np.uint8)

    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min
    #print(w,h)

    s = max(w,h)
    norm_point = (point-[x_min,y_min])/s
    norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)


    #--------
    T = time.max()+1
    P_part1 = 0
    for t in range(int(T/2)):
        P_part1 += len(norm_point[time==t])
    if P_part1 > 0:
	    colors = plt.cm.jet(np.arange(0,P_part1+1)/(P_part1))
	    colors = (colors[:,:3]*255).astype(np.uint8)
    p_num = 0
    for t in range(int(T/2)):
        p = norm_point[time==t]
        x,y = p.T
        image1[y,x]=255
        N = len(p)
        for i in range(N-1):
            color = colors[p_num]
            color = [int(color[2]),int(color[1]),int(color[0])]
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image1,(x0,y0),(x1,y1),color,1,cv2.LINE_AA)
            p_num += 1


    P_part2 = 0
    for t in range(int(T/2),T):
        P_part2 += len(norm_point[time==t])
    if P_part2 > 0:
	    colors = plt.cm.jet(np.arange(0,P_part2+1)/(P_part2))
	    colors = (colors[:,:3]*255).astype(np.uint8)
    p_num = 0
    for t in range(int(T/2),T):
        p = norm_point[time==t]
        x,y = p.T
        image2[y,x]=255
        N = len(p)
        for i in range(N-1):
            color = colors[p_num]
            color = [int(color[2]),int(color[1]),int(color[0])]
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image2,(x0,y0),(x1,y1),color,1,cv2.LINE_AA)
            p_num += 1
    
    image = np.concatenate((image1,image2),axis=2)
            
    image = seq.augment_image(image)

    return image


class DoodleDataset(Dataset):

    def __init__(self, mode, split='<NIL>', augment = null_augment, complexity = 'simplified'):
        super(DoodleDataset, self).__init__()
        assert complexity in ['simplified', 'raw']
        start = timer()

        self.split      = split
        self.augment    = augment
        self.mode       = mode
        self.complexity = complexity

        self.df     = []
        self.id     = []

        if mode=='train':
            TRAIN_DF = []
            # countrycode, drawing, key_id, recognized, timestamp, word

            if TRAIN_DF == []:
                for l,name in enumerate(CLASS_NAME):
                    print('\r\t load df   :  %3d/%3d %24s  %s'%(l,NUM_CLASS,name,time_to_str((timer() - start),'sec')),end='',flush=True)
                    name = name.replace('_', ' ')

                    df = pd.read_csv(DATA_DIR + '/split/%s/%s.csv'%(split,name))
                    TRAIN_DF.append(df)
                print('')
            self.df = TRAIN_DF

            for l,name in enumerate(CLASS_NAME):
                print('\r\t load split:  %3d/%3d %24s  %s'%(l,NUM_CLASS,name,time_to_str((timer() - start),'sec')),end='',flush=True)
                name = name.replace('_', ' ')

                df = TRAIN_DF[l]
                #key_id = np.loadtxt(DATA_DIR + '/split/%s/%s'%(split,name), np.int64)
                key_id = np.load(DATA_DIR + '/split/%s/%s.npy'%(split,name))
                label = np.full(len(key_id),l,np.int64)
                drawing_id = df.loc[df['key_id'].isin(key_id)].index.values
                self.id.append(
                    np.vstack([label, drawing_id, key_id]).T
                )
            self.id = np.concatenate(self.id)
            print('')

        if mode=='test':
            global TEST_DF
            # key_id, countrycode, drawing

            #if TEST_DF == []:
            TEST_DF = pd.read_csv(DATA_DIR + '/csv/test_%s.csv'%(complexity))
            self.id = np.arange(0,len(TEST_DF))

            self.df = TEST_DF

        print('')

    def __str__(self):
        N = len(self.id)
        string = ''\
        + '\tsplit        = %s\n'%self.split \
        + '\tmode         = %s\n'%self.mode \
        + '\tcomplexity   = %s\n'%self.complexity \
        + '\tlen(self.id) = %d\n'%N \
        + '\n'
        return string


    def __getitem__(self, index):


        if self.mode=='train':
            label, drawing_id, key_id = self.id[index]
            drawing = self.df[label]['drawing'][drawing_id]
            drawing = eval(drawing)

        if self.mode=='test':
            label=None
            drawing = self.df['drawing'][index]
            drawing = eval(drawing)

        return self.augment(drawing, label, index)

    def __len__(self):
        return len(self.id)









# check #################################################################
def run_check_train_data():

    dataset = DoodleDataset('train', 'train_0')
    print(dataset)

    #--
    num = len(dataset)
    for m in range(num):
        #i = m
        i = np.random.choice(num)
        image, label, cache = dataset[i]

        print('%8d  %8d :  %3d    %s'%(i,cache.index,label,CLASS_NAME[label]))

        overlay=255-image
        image_show('overlay',overlay, resize=2)
        cv2.waitKey(0)



def run_check_test_data():

    dataset = DoodleDataset('test')
    print(dataset)

    #--
    num = len(dataset)
    for m in range(num):
        i = m
        #i = np.random.choice(num)
        image, label, cache = dataset[i]

        print('%8d  %8d : '%(i,cache.index))

        overlay=255-image
        image_show('overlay',overlay, resize=2)
        cv2.waitKey(0)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_data()
    #run_check_test_data()

