import random

import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

__all__ = ['Logger']

class Logger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)

    def image_summary(self, epoch, tag, image):
        with self.writer.as_default():
            tf.summary.image(f'{epoch}/{tag}', image, max_outputs=1, step=1)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.histogram(name=tag,data=hist.bucket_limit, step=step)
            self.writer.flush()