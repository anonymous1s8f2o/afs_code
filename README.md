# afs_code

**train feature extractors**\
python train.py --pert 3\
python train.py --pert 4\
python train.py --pert 5\
python train.py --pert 6\
python train.py --pert 7\
python train.py --pert 8\
**train linear mergers**\
python train_feature_brother.py --root_path exp/ratio/ratio_0.5    --weights 0 0 0 1 1 1 1 1 1 --ratio 0.5\
**test accuracy and robustness**\
python ratio_exp.py



