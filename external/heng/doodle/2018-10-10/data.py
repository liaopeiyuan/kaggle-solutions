from common import *



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


#------------------------------------------------------------------------

def null_image_augment(point,label,index):
    cache = Struct(point = point.copy(), label = label, index=index)
    image = point_to_image(point, 64, 64)
    return image, label, cache

def null_image_collate(batch):
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





def null_stroke_augment(point,label,index):
    cache = Struct(point = point.copy(), label = label, index=index)
    stroke = point_to_stroke(point)
    return stroke, label, cache



#----------------------------------------
def normalise_point(point):

    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min
    s = max(w,h)

    point[:,:2] = (point[:,:2]-[x_min,y_min])/s
    point[:,:2] = (point[:,:2]-[w/s*0.5,h/s*0.5])

    return point


def point_to_image(point, H, W, border=0.05):

    point = normalise_point(point)
    time  = point[:,2].astype(np.int32)
    norm_point = point[:,:2]
    norm_point = norm_point[:,:2]* max(W,H)*(1-2*border)
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)

    #--------
    image = np.zeros((H,W,3),np.uint8)

    T = time.max()+1
    for t in range(T):
        p = norm_point[time==t]
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image,(x0,y0),(x1,y1),(255,255,255),1,cv2.LINE_AA)

        x,y = p.T
        image[y,x]=(255,255,255)
    return image



def point_to_stroke(point):

    point = normalise_point(point)
    num_point = len(point)
    #stroke =[dx,dy,dt]

    #--------
    stroke    = np.zeros((num_point,3),np.float32)
    stroke[0] = [0,0,1]
    stroke[1:] = point[1:]- point[:-1]

    return stroke


def draw_point_to_overlay(point):
    H, W   = 256, 256
    border = 0.05

    point = normalise_point(point)
    time  = point[:,2].astype(np.int32)
    norm_point = point[:,:2]
    norm_point = norm_point[:,:2]* max(W,H)*(1-2*border)
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)

    #--------
    overlay = np.full((H,W,3),255,np.uint8)


    T = time.max()+1
    colors = plt.cm.jet(np.arange(0,T)/(T-1))
    colors = (colors[:,:3]*255).astype(np.uint8)

    for t in range(T):
        color = colors[t]
        color = [int(color[2]),int(color[1]),int(color[0])]

        p = norm_point[time==t]
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i  ]
            x1,y1 = p[i+1]
            cv2.line(overlay,(x0,y0),(x1,y1),color,2,cv2.LINE_AA)

        for i in range(N):
            x,y = p[i]
            #cv2.circle(overlay, (x,y), 4, [255,255,255], -1, cv2.LINE_AA)
            cv2.circle(overlay, (x,y), 4, [0,0,0], -1, cv2.LINE_AA)
            cv2.circle(overlay, (x,y), 3, color, -1, cv2.LINE_AA)



        # x,y = p.T
        # overlay[y,x]=(255,255,255)
    return overlay


##--------------------------------------------------------------------------------
def read_one_df_file(df_file):
    name = df_file.split('/')[-1].replace('.csv','')
    print('\r\t load df : %16s '%(name),end='',flush=True)
    df = pd.read_csv(df_file)
    return df

class DoodleDataset(Dataset):

    def __init__(self, mode, split='<NIL>', augment = null_image_augment, complexity = 'simplified'):
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
            global TRAIN_DF
            # countrycode, drawing, key_id, recognized, timestamp, word

            if TRAIN_DF == []:
                df_files =[
                    DATA_DIR + '/csv/train_%s/%s.csv'%(complexity,name.replace('_', ' ')) for name in CLASS_NAME
                ]

                pool = Pool(processes=16)
                TRAIN_DF = pool.map(read_one_df_file, df_files)
                pool.close()
                pool.join()

                #TRAIN_DF = [read_one_df_file(f) for f in df_files]

            self.df = TRAIN_DF

            for l,name in enumerate(CLASS_NAME):
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

            if TEST_DF == []:
                TEST_DF = pd.read_csv(DATA_DIR + '/csv/test_%s.csv'%(complexity))
                self.id = np.arange(0,len(TEST_DF))

            self.df = TEST_DF

        print('\r\t load dataset: %s'%(time_to_str((timer() - start),'sec')))
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

        point = []
        for t,(x,y) in enumerate(drawing):
            point.append(np.array((x,y,np.full(len(x),t)),np.float32).T)
        point = np.concatenate(point)

        return self.augment(point, label, index)

    def __len__(self):
        return len(self.id)









# check #################################################################
def run_check_train_image_data():

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



def run_check_test_image_data():

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


def run_check_train_stroke_data():

    dataset = DoodleDataset('train', 'train_0', augment=null_stroke_augment)
    print(dataset)

    #--
    num = len(dataset)
    for m in range(num):
        #i = m
        i = np.random.choice(num)
        stroke, label, cache = dataset[i]

        print('%8d  %8d :  %3d    %s'%(i,cache.index,label,CLASS_NAME[label]))


        overlay = draw_point_to_overlay(cache.point)
        draw_shadow_text(overlay,'%3d %s'%(label,CLASS_NAME[label]),(3,20),0.6,[255,255,255],1)


        image_show('overlay',overlay, resize=1)
        cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_stroke_data()


