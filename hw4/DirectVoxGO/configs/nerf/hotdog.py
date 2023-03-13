_base_ = '../default.py'

expname = 'trybest'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/mnt/sda/DLCV2/hw4_data/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)

