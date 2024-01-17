from model.unet import Unet
import tensorflow as tf

model = Unet()
#model = Unet(tf.zeros((1,572,572,1)))
#model.summary()
print(Unet())
print(model.shape)

for layer in model(tf.zeros((1,572,572,1))).layers:
    print(layer.output_shape)
