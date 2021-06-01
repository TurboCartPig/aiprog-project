import os

import cv2
import numpy as np

# img1 = cv2.imread("datasets/leedsbutterfly/images/Danaus plexippus/0010001.png")
# img2 = cv2.imread(
#     "datasets/leedsbutterfly/segmentations/Danaus plexippus/0010001_seg0.png"
# )

# aaaa = cv2.bitwise_and(img1, img2, mask=None)

# cv2.imshow("CV2", aaaa)

# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()

images = os.walk("./datasets/leedsbutterfly/images/")
masks = os.walk("./datasets/leedsbutterfly/segmentations/")

pairs = zip(images, masks)

for pair in pairs:
    (dirpath1, dirnames1, _), (dirpath2, dirnames2, _) = pair
    for imagedir, maskdir in zip(dirnames1, dirnames2):
        print(imagedir)
        images = os.walk(os.path.join(dirpath1, imagedir))
        masks = os.walk(os.path.join(dirpath2, maskdir))
        for imageclass, maskclass in zip(images, masks):
            dirpatha1, _, filenames1 = imageclass
            dirpatha2, _, filenames2 = maskclass
            print(dirpatha1)
            for image, mask in zip(filenames1, filenames2):
                print(os.path.join(dirpatha1, image))
                img1 = cv2.imread(os.path.join(dirpatha1, image))
                img2 = cv2.imread(os.path.join(dirpatha2, mask))
                gen = cv2.bitwise_and(img1, img2, mask=None)
                cv2.imwrite(os.path.join("./datasets/gen", imagedir, image), gen)
