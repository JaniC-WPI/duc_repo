import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('/home/jc-merlab/lama/predict_data/08_14_2023/raw_and_mask/254.png')     # input
mask = cv2.imread('/home/jc-merlab/lama/predict_data/08_14_2023/raw_and_mask/254_mask.png',0)  # mask

dst_TELEA = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
dst_NS = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

plt.subplot(221), plt.imshow(img)
plt.title('degraded image')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.title('mask image')
plt.subplot(223), plt.imshow(dst_TELEA)
plt.title('TELEA')
plt.subplot(224), plt.imshow(dst_NS)
plt.title('NS')

plt.tight_layout()
plt.show()