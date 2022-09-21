
import albumentations as A
import cv2

class YoloAugmentations:
    
    def __init__(self, parameters = {
        'to_float': {'max_value': 65535.0},
        'random_brightness_contrast': { 'p': 0.5,},
        'random_gamma': { 'p': 0.5 },
        'sharpen': {'p': 0.5},
        'random_horizontal_flip': { 'p': 0.5 },
        'random_vertical_flip': { 'p': 0.5 },
        'shift_scale_rotate': {'rotate_limit': 5, 'interpolation' : cv2.INTER_NEAREST , 'border_mode': cv2.BORDER_CONSTANT},
        'from_float': {'max_value': 65535.0}
    }, bbox_format='yolo', img_size=(256, 800)):
        self.parameters = parameters
        self.transform = A.Compose([
                A.ToFloat(**parameters['to_float']),
                #A.BBoxSafeRandomCrop(always_apply=True),
                A.RandomBrightnessContrast(**parameters['random_brightness_contrast']),
                A.RandomGamma(**parameters['random_gamma']),
                #A.Sharpen(**parameters['sharpen']),
                A.OneOf([
                    A.RandomSizedBBoxSafeCrop(height=img_size[0], width=img_size[1],
                                              always_apply=True, interpolation=cv2.INTER_LINEAR),
                    A.Resize(height=img_size[0], width=img_size[1],always_apply=True, interpolation=cv2.INTER_LINEAR),
                ], p=1.0),
                A.HorizontalFlip(**parameters['random_horizontal_flip']),
                A.VerticalFlip(**parameters['random_vertical_flip']),
                A.ShiftScaleRotate(**parameters['shift_scale_rotate']),
                A.FromFloat(**parameters['from_float']),
            ], bbox_params=A.BboxParams(format=bbox_format))
        
        self.img_size = img_size
    
    def __call__(self, datapoint):
        # datapoint in a dict with keys = {'image', 'bboxes'}
        return self.transform(image=datapoint['image'], bboxes=datapoint['bboxes'])