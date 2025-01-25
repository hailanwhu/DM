import optparse
import os

def parse_arg(istest=2):
    parser = optparse.OptionParser()
    
    parser.add_option('--cords_fnm', dest='cords_fnm', default='tmp/cords.log', type='str')
    parser.add_option('--input_fnm', dest='input_fnm', default='test/demo_query.txt', type='str')
    parser.add_option('--data_dir', dest='data_dir', default='data', type='str')
    parser.add_option('--output_dir', dest='output_dir', default='res', type='str')
    parser.add_option('--model_fnm', dest='model_fnm', default='model', type='str')

    parser.add_option('--run_cords', dest="run_cords", help="Run cords or use cache. Default: 1", default=1, type='int')
    parser.add_option('--extract_emb', dest="extract_emb", help="Extract embedding vectors from input model or use cache. Default: 1", default=1, type='int')

    parser.add_option('--input_rate', dest="input_rate", help="Input dataset size. Default: 1.0", default=1, type='float')
    parser.add_option('--sample_size', dest="sample_size", help="Sample Size. Default: 128", default=128, type='int')
    parser.add_option('--nusecpp', dest="nusecpp", help="Use CPP. Default: 0", default=0, type='int')
    parser.add_option('--nusebucket', dest="nusebucket", help="Use prior buckets. Default: ''(empty)", default='', type='str')

    parser.add_option('--nn', dest="nn", help="# FC layers. Default: 0.", default=0, type='int')
    parser.add_option('--nfc', dest="nfc", help="FC layers size. Default: 128.", default=128, type='int')
    parser.add_option('--nr', dest="nr", help="Embedding size. Default: 128.", default=128, type='int')

    parser.add_option('--storage', dest="storage", help="Storage used (X 4KB/col) including overhead", default=1, type='float')
    parser.add_option('--max_atom_budget', dest="max_atom_budget", help="max atom budget (# floats)", default=512, type='int')

    parser.add_option('--nm', dest="nm", help="Training compression rate. Default: 0.5.", default=0.5, type='float')
    parser.add_option('--nt', dest="nt", help="Total cell embedding budget. Default: 2048.", default=2048, type='int')
    parser.add_option('--nc', dest="nc", help="Total column set number budget. Default: 2.", default=2, type='int')
    parser.add_option('--nx', dest="nx", help="Total embedding number budget. Default: 4.", default=4, type='int')

    parser.add_option('--neb', dest="neb", help="Max column resolution: 128.", default=128, type='int')

    parser.add_option('--nlen', dest="nlen", help="Input length. Default: 2000000.", default=2000000, type='int')
    parser.add_option('--normlen', dest="normlen", help="Norm length. Default: 2000000.", default=2000000, type='int')

    parser.add_option('--mind', dest='mind', help='Minimum sketch dimensions. Defalut: 2', default=2, type='int')
    parser.add_option('--maxd', dest='maxd', help='Maximum sketch dimensions. Defalut: 2', default=2, type='int')
    parser.add_option('--nb', dest="nb", help="Base size. Default: 2.", default=2, type='int')

    # training parameters
    parser.add_option('--nlr', dest="nlr", help="Learning rate. Default: 1e-2.", default=1e-2, type='float')
    parser.add_option('--ngpus', dest="ngpus", help="# GPUs. Default: 1.", default=1, type='int')
    parser.add_option('--nl', dest="nl", help="Training pool size. Default: 1x.", default=1, type='float')
    parser.add_option('--nbat', dest="nbat", help="Batch size. Default: 2048.", default=2048, type='int')
    parser.add_option('--nst', dest="nst", help="# of steps per epoch: 1280.", default=1280, type='int')
    parser.add_option('--nep', dest="nep", help="# of Epoches. Default: 64.", default=64, type='int')
    parser.add_option('--ne', dest="ne", help="Individual cell embedding length. Default: 64.", default=64, type='int')
    parser.add_option('--verbose', dest="nverbose", help="verbose.", default=1, type='int')

    options, args = parser.parse_args()
    options.ncd = options.nt - (int)(options.nt * (1 - options.nm))
    return options
