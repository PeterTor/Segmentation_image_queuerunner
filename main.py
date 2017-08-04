from tensorpack import *
import filereader
import dataset
import numpy as np
import time

reader = dataset.cityscapes_filereader()
image_pairs = reader.get_image_pairs()
ds = filereader.ImageFromFile(image_pairs, channel=3, shuffle=True)
ds = BatchData(ds, 4) # use batchsize = 4
ds = PrefetchData(ds, 100) # use queue size 100
ds = PrefetchDataZMQ(ds,4) # use 4 threads
ds.reset_state()

# do stuff
t = time.time()
for dp in ds.get_data():
    print np.shape(dp)
    print time.time() - t
    t = time.time()