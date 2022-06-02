import os

from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

gan_generator = generator()
gan_generator.load_weights(weights_file('pre_generator.h5'))


### MSE

# gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator(), new_loss = 'MSE', loss_n=100)
# gan_trainer.train(train_ds, steps=200000)

# gan_trainer.generator.save_weights(weights_file('gan_generator_mse_100.h5'))
# gan_trainer.discriminator.save_weights(weights_file('gan_discriminator_mse_100.h5'))


### Weighted MSE

gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator(), new_loss = 'W-MSE', loss_n=100)
gan_trainer.train(train_ds, steps=200000)

gan_trainer.generator.save_weights(weights_file('gan_generator_wmse_100.h5'))
gan_trainer.discriminator.save_weights(weights_file('gan_discriminator_wmse_100.h5'))