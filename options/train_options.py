from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self) -> None:
        ## Method HyperParams
        self.parser.add_argument('--h_ratio', type=float, default=0.55, help='the ratio of upper parts')
        self.parser.add_argument('--w_ratio', type=float, default=0.5, help='the ratio of left parts')
        self.parser.add_argument('--no_global', action='store_true', help='no global G')
        self.parser.add_argument('--no_local', action='store_true', help='no local Gs')
        self.parser.add_argument('--architecture', default='SE', help='SE|DE|TE for local generator')
        self.parser.add_argument('--all_D', action='store_true', help='using all 3 Discriminators in SE|DE|TE instead of 3|2|1')
        self.parser.add_argument('--final_parts', action='store_true', help='using final parts for AdvLoss')
        self.parser.add_argument('--ld_layer', type=int, default=4, help='num_layer for local G down')
        self.parser.add_argument('--cb_layer', type=int, default=2, help='num_layer for G combiner')
        
        ## Dataset Path
        self.parser.add_argument('--data_folder', default='../Datasets/My-CUFS-New/', help='the dataset folder')
        self.parser.add_argument('--train_sketch_list', default='files/train/list_sketch.txt', help='the list of train sketches in dataset')
        self.parser.add_argument('--train_photo_list', default='files/train/list_photo.txt', help='the list of train photos in dataset')
        self.parser.add_argument('--test_sketch_list', default='files/test/list_sketch.txt', help='the list of test sketches in dataset')
        self.parser.add_argument('--test_photo_list', default='files/test/list_photo.txt', help='the list of test photos in dataset')
        self.parser.add_argument('--fid_list', default='files/list_fid.txt', help='the list of label folders for FID')
        
        ## Data Description
        self.parser.add_argument('--input_nc', type=int, default=1, help='number of input sketch channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='number of output photo channels')
        self.parser.add_argument('--pad', action='store_true', help='whether to zeropad to target size')
        self.parser.add_argument('--output_shape', type=int, default=256, help='shape of output photo')
        
        ## Model Path
        self.parser.add_argument('--vgg_model', default='../Models/vgg.model', help='path of VGG model for Perceptual Loss')
        self.parser.add_argument('--inception_model', default='../Models/pt_inception.pth', help='path of Inception model for FID')
        
        ## Loss Params
        self.parser.add_argument('--use_BCE', action='store_true', help='use BCELoss for Adversarial Loss, and Sigmoid for D')
        self.parser.add_argument('--pxl_loss', default='L1', help='L1|L2 for pixel matching loss')
        self.parser.add_argument('--no_perceptual', action='store_true', help='not using Perceptual Loss')
        self.parser.add_argument('--alpha', type=float, default=0.7, help='weight of Global L1 Loss in Compositional Loss')
        self.parser.add_argument('--vgg_layers', type=int, default=3, help='layers of VGG to use for Perceptual Loss')
        
        ## Optim Params
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of all optimizers')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of all Adam optimizers')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of all Adam optimizers')
        self.parser.add_argument('--delta', type=float, default=1, help='weight of Adversarial Loss')
        self.parser.add_argument('--lamda', type=float, default=2.5, help='weight of Compositional Loss')
        self.parser.add_argument('--gamma', type=float, default=10, help='weight of Perceptual Loss')
        
        ## Train Options
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
        self.parser.add_argument('--batch_size', type=int, default=4, help='dataloader batch size')
        
        ## Test Options
        self.parser.add_argument('--test_start', type=int, default=50, help='which epoch to start test')
        self.parser.add_argument('--test_period', type=int, default=50, help='how many epochs to test once')
        # self.parser.add_argument('--save_image_when_test', action='store_true', help='whether to save gen images after test')
        self.parser.add_argument('--save_models', action='store_true', help='whether to save models after test')
        self.parser.add_argument('--train_show_list', type=list, default=[1, 91, 175], help='which train images to show')
        self.parser.add_argument('--test_show_list', type=list, default=[1, 101, 144], help='which test images to show')
        
        ## Logs and Saves
        self.parser.add_argument('--logs_folder', default='./Saves/Logs', help='logs folder for TensorBoard')
        self.parser.add_argument('--log_name', default='exp_debug', help='log name of current run')
        self.parser.add_argument('--model_saves_folder', default='./Saves/Checkpoints', help='saves folder for model')
        self.parser.add_argument('--image_saves_folder', default='./Saves/Images', help='saves folder for image')
