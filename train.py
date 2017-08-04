import tensorflow as tf
import model as m
import matplotlib.pyplot as plt
import numpy as np
from lib import get_dataset
from lib.get_dataset import get_iterator
import pspnet
from os import walk

NUM_CLASSES = 7 #6 plus 0
BATCH_SIZE = 32
WINDOW_LENGTH = 8
w = WINDOW_LENGTH

def get_last_train_file():
    f = []
    for (dirpath, dirnames, filenames) in walk("modelfiles/"):
        f.extend(filenames)
        break
    return int(str(f[-1]).split(".")[0][:])

def generate_output(r,sess,image,annotation_pred,keep_probabilty,name):
    picture = []
    for i in np.arange(0,5200,w):
        input_i = r.get_data_at(i,w)
        feed_dict = {image:input_i, keep_probabilty:1.0}
        pred_i = sess.run(annotation_pred,feed_dict)
        picture.append(pred_i)
    picture=np.concatenate(picture,axis=2)
    picture=np.squeeze(picture)
    plt.imshow(picture)
    plt.savefig('output/'+str(name)+'.png')
    plt.close()


#nikon = 4256x2832
#manta = 2452x2056
def main():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None,512,1024,3], name="input_image")
    is_training = tf.placeholder(tf.bool,None)

    annotation = tf.placeholder(tf.int32, shape=[None,512,1024,20], name="annotation")

    #logits,annotation_pred = m.create_network(image, keep_probability,w,NUM_CLASSES)
    logits, endpoints = pspnet.pspnet_v1_50(image,num_classes=21,is_training=is_training)
    pred_annotation = tf.argmax(logits,axis=3)

    #exclude unkown class 0
    _labels = tf.argmax(annotation,axis=3)
    _index = tf.where(tf.not_equal(_labels, tf.constant(0, dtype=tf.int64)))
    _logits = tf.gather_nd(logits, _index)
    _labels = tf.gather_nd(_labels, _index)

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits,labels=_labels)))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    #c_iter = get_last_train_file()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "modelfiles/m-95200")
    D = get_dataset.get_dataset("cityscapes", "train",folder="/media/peters/Data/cityscapes/sem_seg2")
    train_iter = get_iterator(D, batch_size=4,multi_worker=False,num_worker=2)

    D2 = get_dataset.get_dataset("cityscapes", "valid",folder="/media/peters/Data/cityscapes/sem_seg2")
    valid_iter = get_iterator(D2, batch_size=4,multi_worker=False,num_worker=2)

    coarse = get_iterator(get_dataset.get_dataset("cityscapes", "train_extra",folder="/media/peters/Data/cityscapes/sem_seg2"), batch_size=4, multi_worker=False,num_worker=2)

   # D1 = get_dataset.get_dataset("cityscapes", "test", folder="/media/peters/Data/cityscapes/sem_seg2")
   # train_test_iter = get_iterator(D1, batch_size=1, multi_worker=False)

    counter = 95200
    while True:
        if counter % 5 == 0:
            batch = train_iter.next()
        else:
            batch = coarse.next()

        im_batch = batch["input"]
        label_batch = batch["label_sem_seg"]
        feed_dict = {image: im_batch, annotation: label_batch, keep_probability: 0.85, is_training:True}
        sess.run(optimizer, feed_dict=feed_dict)

        if counter % 10 == 0:
            batch =  valid_iter.next()
            im_batch = batch["input"]
            label_batch = batch["label_sem_seg"]

            feed_dict = {image: im_batch, annotation: label_batch, keep_probability: 1.0, is_training: False}
            print "ep:", counter," loss:",sess.run(loss,feed_dict)

        if counter % 50 == 0:
            feed_dict = {image: im_batch, annotation: label_batch, keep_probability: 1.0, is_training: False}
            anon = sess.run(pred_annotation,feed_dict)
            plt.imshow(anon[0])
            plt.savefig("output/"+str(counter)+".png")
            plt.close()
            plt.imshow(np.argmax(label_batch[0],axis=2))
            plt.savefig("output/"+str(counter)+"_gt.png")
            plt.close()

        if counter % 100 == 0 and counter > 95200:
            print "saved model ",counter
            saver.save(sess,"modelfiles/m",global_step=counter)
        counter += 1




main()