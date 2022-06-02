## weights

- pre_generator.h5 : pre-trained weights (Use this for fine-grained GAN training)
- gan_discriminator.h5 / gan_generator.h5 : Original SRGAN model weights
- gan*discriminator_mse*{n}.h5 / gan*generator_mse*{n}.h5 : SRGAN with new loss function = (VGG loss) + 1/n \* (Pixel-wise loss)
