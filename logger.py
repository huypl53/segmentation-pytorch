from io import BytesIO

import scipy.misc
from torch.utils.tensorboard import SummaryWriter


# TODO: put writer.close()

class Logger(object):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, image, step):
        self.writer.add_image(tag, image, step)

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        self.writer.add_images("{}".format(tag), images, step)
