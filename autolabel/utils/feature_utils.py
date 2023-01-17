def get_feature_extractor(features, checkpoint=None):
    if features == 'fcn50':
        from autolabel.features import FCN50
        return FCN50()
    elif features == 'dino':
        from autolabel.features import Dino
        return Dino()
    elif features == 'lseg':
        from autolabel.features import lseg
        return lseg.LSegFE(checkpoint)
    else:
        raise NotImplementedError()
