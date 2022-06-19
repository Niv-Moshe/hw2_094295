from imgaug import augmenters as iaa

affine_transformation = dict(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                             translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                             rotate=(-20, 20),
                             shear=(-3, 3),
                             cval=255)  # filling with white pixels
# Labels to perform no flip: iv, vi, vii, viii
transform_no_flip = iaa.Sequential([iaa.Resize({"height": 400, "width": 400}),
                                    iaa.Crop(percent=(0, 0.05)),
                                    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                                    iaa.LinearContrast((1.35, 1.75)),
                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                                    iaa.Affine(**affine_transformation)], random_order=False)

# Labels to perform only horizontal flip: v
transform_horizontal = iaa.Sequential([iaa.Fliplr(0.5),  # horizontal
                                       iaa.Resize({"height": 400, "width": 400}),
                                       iaa.Crop(percent=(0, 0.05)),
                                       iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                                       iaa.LinearContrast((1.35, 1.75)),
                                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                       iaa.Multiply((0.8, 1.2), per_channel=0.2),
                                       iaa.Affine(**affine_transformation)], random_order=False)

# Labels to perform only vertical flip: ix
transform_vertical = iaa.Sequential([iaa.Flipud(0.5),  # vertical
                                     iaa.Resize({"height": 400, "width": 400}),
                                     iaa.Crop(percent=(0, 0.05)),
                                     iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                                     iaa.LinearContrast((1.35, 1.75)),
                                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                     iaa.Multiply((0.8, 1.2), per_channel=0.2),
                                     iaa.Affine(**affine_transformation)], random_order=False)

# Labels to perform horizontal and vertical flip: i, ii, iii, x
transform_horizontal_vertical = iaa.Sequential([iaa.Flipud(0.5),  # vertical
                                                iaa.Fliplr(0.5),  # horizontal
                                                iaa.Resize({"height": 400, "width": 400}),
                                                iaa.Crop(percent=(0, 0.05)),
                                                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                                                iaa.LinearContrast((1.35, 1.75)),
                                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                                                iaa.Affine(**affine_transformation)], random_order=False)
