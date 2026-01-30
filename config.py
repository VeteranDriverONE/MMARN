import argparse as arg

parser = arg.ArgumentParser()

# data organization parameters
parser.add_argument('--train_dir',default="E:\\datasets\\kidney_processed\\128_128_64", help = 'training files')
parser.add_argument('--model_dir',default='./checkpoints',help="model output directory")
parser.add_argument('--load_model',default=None,help="optional model file to initialize with")
parser.add_argument('--load_Trans',default=None,help="optional model file to initialize with")
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default='./Result')
parser.add_argument('--train_tb', default='./img/train')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
parser.add_argument('--device', action='store_true', help='specify that data has multiple channels')
# parser.add_argument('--types', type=tuple, default=('CMP','NP','UP'),help='recursive_cascade_level')
parser.add_argument('--types', type=tuple, default=('t1','t21','t1ce1'),help='recursive_cascade_level')
# parser.add_argument('--types', type=tuple, default=('T1DUAL','T2SPIR'),help='recursive_cascade_level')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--n_cas', default=3,help='recursive_cascade_level')
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epoch',default=500 , type= int,help='number of training epoch')
# parser.add_argument('--img_shape',default=(64, 128, 128))
parser.add_argument('--img_shape',default=(64, 64,64))
parser.add_argument('--save_per_epoch', type=int, default=50, help='frequency of model saves (default: 100)')
parser.add_argument('--tb_save_freq', type=int, default=400, help='frequency of tensorboard saves (default: 100)')
parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--adam', type=bool, default=True, help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

# loss hyperparameters
parser.add_argument('--image_loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')
parser.add_argument('--auto_set_weights', default=True, type=bool, help='自动权重设定')

# test
# parser.add_argument('--test_load_model', default='E:\\workspace\\Python\\PARNet6\\checkpoints\\parnet_0200.pt', help="optional model file to initialize with")
parser.add_argument('--test_load_model', default=None, help="optional model file to initialize with")
parser.add_argument('--test_load_trans', default='E:\\workspace\\Python\\PARNet6\\checkpoints\\Trans_encoder_0200.pt', help="optional model file to initialize with")
# parser.add_argument('--test_load_trans_model', default=None, help="optional model file to initialize with")
parser.add_argument('--test_image_dir', default="E:\\datasets\\kidney_processed\\test-PARNet\\moving_img", help = 'test files')
parser.add_argument('--test_label_dir', default="E:\\datasets\\kidney_processed\\test-PARNet\\moving_label")
parser.add_argument('--origin_dir', default="E:\\datasets\\kidney_processed\\reg_origin_test")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--test_tb', default='./img/test')

parser.add_argument('--trans_jpg', default='./train_T_jpg')
parser.add_argument('--test_trans_jpg', default='./test_T_jpg')
parser.add_argument('--test_jpg_outpath', default='./reg_test_img')

parser.add_argument('--root_path', default='E:\\workspace\\Python\\PARNet6')
args = parser.parse_args()