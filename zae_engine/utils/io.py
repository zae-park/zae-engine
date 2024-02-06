from typing import Union
from PIL import Image
import urllib.request
import time

import os

IMAGE_FORMAT = ['png', 'jpg', 'jpeg', 'til']


def image_from_url(url: str, save_dst: str = None) -> Union[None, Image]:

	save_mode = False if save_dst is None else True

	if save_dst is None:
		save_dst = os.path.join(os.getcwd(), str(time.time()).replace(".", "") + ".png")
	else:
		if ext := os.path.splitext(save_dst)[-1] not in IMAGE_FORMAT:
			raise AssertionError(f'Invalid extension. Expect one of {IMAGE_FORMAT}, but receive "{ext}".')
	urllib.request.urlretrieve(url, save_dst)
	if save_mode:
		return
	else:
		img = Image.open(save_dst)
		os.remove(save_dst)
		return img

