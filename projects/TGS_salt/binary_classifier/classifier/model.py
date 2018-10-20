import torch.nn as nn
import pretrainedmodels


class classifier(nn.Module):
    def __init__(self, model_name='resnet32'):
        super(classifier, self).__init__()

        # Load pretrained ImageNet model
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        
        print(model_name + ' model settings:')
        for var in pretrainedmodels.pretrained_settings[model_name]['imagenet']:
            print('\t' + var + ': '+ str(pretrainedmodels.pretrained_settings[model_name]['imagenet'][var]))
            
        # Define last layer for fine-tuning
        dim_feats = self.model.last_linear.in_features
        nb_classes = 1
        #self.model.last_linear = nn.Linear(dim_feats, nb_classes)
        self.model.last_linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(dim_feats, nb_classes))

    def forward(self, input):
        return self.model(input)

    def set_mode(self, mode):
        self.mode = mode

        if 'validation' in mode or 'test' in mode:
            self.eval()

        elif 'train' in mode:
            self.train()

        else:   
            raise NotImplementedError
