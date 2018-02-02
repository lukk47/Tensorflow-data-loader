# Tensorflow-data-loader
Reading data into tensorflow using tf.data function

# Example of usage and testing
```
import matplotlib.pyplot as plt
import tensorflow as tf
import data_loader

data_list = '/media/lok/dataset/lok/project/Enhanceenet/list/trains2.txt'
plt.ioff()

Parse the images and masks, and return the data in batches, augmented optionally
data, init_op = data_loader.data_batch(data_list, augment=['flip_ud','flip_lr','rot90'], normalize=True,batch_size=20, epoch = None)
Get the image and mask op from the returned dataset
image_tensor, mask_tensor = data

with tf.Session() as sess:
    sess.run(init_op)
    # Evaluate the tensors
    for i in range(1):
        image, mask = sess.run([image_tensor, mask_tensor])

        # Confirming everything is working by visualizing
        plt.figure('augmented image')
        plt.imshow(image[0, :, :, :])
        plt.figure('augmented mask')
        plt.imshow(mask[0, :, :,:])
        plt.show()
    # Do whatever you want now, like creating a feed dict and train your models
```

# Reference
