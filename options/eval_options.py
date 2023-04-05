from .base_options import BaseOptions


class EvalOptions(BaseOptions):
    def initialize(self) -> None:
        ## Method HyperParams
        self.parser.add_argument('--h_ratio', type=float, default=0.55, help='the ratio of upper parts')
        self.parser.add_argument('--w_ratio', type=float, default=0.5, help='the ratio of left parts')
        self.parser.add_argument('--architecture', default='SE', help='SE|DE|TE for local generator')
        
        # Metric
        self.parser.add_argument('--metric', default='fid', help='choose in <fid|fsim|speed>')
        
        ## Data Description
        self.parser.add_argument('--input_nc', type=int, default=1, help='number of input sketch channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='number of output photo channels')
        
        ## Dataset Path
        self.parser.add_argument('--pred_folder', default='./Saves/Images/debug/Test/400', help='the pred folder')
        self.parser.add_argument('--data_folder', default='../Datasets/My-CUFS-New/', help='the dataset root folder')
        self.parser.add_argument('--train_sketch_list', default='files/train/list_sketch.txt', help='the list of train sketches in dataset')
        self.parser.add_argument('--train_photo_list', default='files/train/list_photo.txt', help='the list of train photos in dataset')
        self.parser.add_argument('--test_sketch_list', default='files/test/list_sketch.txt', help='the list of test sketches in dataset')
        self.parser.add_argument('--test_photo_list', default='files/test/list_photo.txt', help='the list of test photos in dataset')
        self.parser.add_argument('--fid_list', default='files/list_fid.txt', help='the list of label folders for FID')
        
        ## Model Path
        self.parser.add_argument('--inception_model', default='../Models/pt_inception.pth', help='path of Inception model for FID')
        
