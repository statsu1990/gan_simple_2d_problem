import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam

# 今更きけないGAN
# https://qiita.com/triwave33/items/1890ccc71fab6cbca87e
# https://qiita.com/pacifinapacific/items/6811b711eee1a5ebbb03

class GAN():
    def __init__(self, latent_dim=2, data_dim=2):
        
        # 潜在変数の次元数
        self.latent_dim = latent_dim
        # データの次元
        self.data_dim = data_dim

        return

    def make_model(self, gene_hidden_neurons, disc_hidden_neurons):

        # discriminator model
        self.disc_model = self.__build_discriminator(disc_hidden_neurons)
        #self.disc_model.compile(optimizer=Adam(lr=1e-5, beta_1=0.1), loss='binary_crossentropy', metrics=['accuracy'])
        self.disc_model.compile(optimizer=Adam(lr=5e-6, beta_1=0.1), loss='binary_crossentropy', metrics=['accuracy'])

        # generator model
        self.gene_model = self.__build_generator(gene_hidden_neurons)

        # combined model of generator and discriminator
        self.combined_model = self.__build_combined_gene_and_disc()
        #self.combined_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.5), loss='binary_crossentropy')
        self.combined_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.5), loss='binary_crossentropy')

        return

    def __build_generator(self, hidden_neurons):
        '''
        build generator keras model
        the last activation is tanh.
        '''

        # input
        latent_inputs = Input(shape=(self.latent_dim,))

        # hidden layer
        x = latent_inputs
        for hidden_n in hidden_neurons:
            x = Dense(hidden_n)(x)
            #x = Activation('relu')(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
        
        # output
        x = Dense(self.data_dim)(x)
        datas = Activation('tanh')(x)
        #datas = Activation('linear')(x)

        model = Model(input=latent_inputs, output=datas)
        model.summary()

        return model

    def __build_discriminator(self, hidden_neurons):
        '''
        build discriminator keras model
        '''

        # input
        datas = Input(shape=(self.data_dim,))

        # hidden layer
        x = datas
        for hidden_n in hidden_neurons:
            x = Dense(hidden_n)(x)
            x = Activation('relu')(x)
            #x = LeakyReLU()(x)
            #x = BatchNormalization()(x)
        
        # output
        x = Dense(1)(x)
        real_or_fake = Activation('sigmoid')(x)

        #
        model = Model(input=datas, output=real_or_fake)
        model.summary()

        return model

    def __build_combined_gene_and_disc(self):
        '''
        build combined keras model of generator and discriminator
        '''
        
        # input
        latent_inputs = Input(shape=(self.latent_dim,))

        # data
        data = self.gene_model(latent_inputs)

        # true or false
        self.disc_model.trainable = False
        real_or_fake = self.disc_model(data)

        #
        model = Model(input=latent_inputs, output=real_or_fake)
        model.summary()
        
        return model

    def train(self, real_datas, epoch, batch_size=32):
        '''
        training gan model
        '''
        print('start training gan model')
        for iep in range(epoch):
            #self.train_step(real_datas, batch_size, iep)
            self.train_step_test1(real_datas, batch_size, iep)
        print('end training')

        return

    def train_step(self, real_datas, batch_size=32, now_epoch=None, print_on_batch=False):
        '''
        training gan model on one epoch
        discriminatorの学習時にrealとfakeを別々に学習
        '''
        
        #
        sample_num = real_datas.shape[0]
        half_batch_size = int(batch_size / 2)
        batch_num = int(sample_num / half_batch_size) + 1

        # index for minibatch training
        shuffled_idx = np.random.permutation(sample_num)

        # roop of batch
        for i_batch in range(batch_num):
            if half_batch_size*i_batch < sample_num:
                # ---------------------------
                # training of discriminator
                # ---------------------------

                # real data
                real_x = real_datas[shuffled_idx[half_batch_size*i_batch : half_batch_size*(i_batch+1)]]
                x_num = real_x.shape[0]
                y = np.ones((x_num, 1)) # label = 1
                #
                disc_loss_real = self.disc_model.train_on_batch(x=real_x, y=y)
            
                # fake data
                latents = np.random.normal(0, 1, (x_num, self.latent_dim))
                fake_x = self.gene_model.predict(latents)
                y = np.zeros((x_num, 1)) # label = 0
                #
                disc_loss_fake = self.disc_model.train_on_batch(x=fake_x, y=y)

                # loss
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)
                
                # ---------------------------
                # training of generator
                # ---------------------------

                # generated data
                # batch size = x_num * 2 (= real + fake in disc training)
                latents = np.random.normal(0, 1, (x_num * 2, self.latent_dim)) # x_num * 2
                y = np.ones((x_num * 2, 1)) # label = 1
                #
                gene_loss = self.combined_model.train_on_batch(x=latents, y=y)

                # training progress
                if print_on_batch:
                    print_epoch = now_epoch if now_epoch is not None else 0
                    print ("epoch: %d, batch: %d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (print_epoch, i_batch+1, disc_loss[0], 100*disc_loss[1], gene_loss))
                
        # training progress
        print_epoch = now_epoch if now_epoch is not None else 0
        print ("epoch: %d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (print_epoch, disc_loss[0], 100*disc_loss[1], gene_loss))
        #print ("epoch: %d, [D loss: %f, acc.: %.2f%%]" % (print_epoch, disc_loss[0], 100*disc_loss[1]))

        return

    def train_step_test1(self, real_datas, batch_size=32, now_epoch=None, print_on_batch=False):
        '''
        training gan model on one epoch
        discriminatorの学習時にrealとfakeを一緒に学習
        '''

        #
        sample_num = real_datas.shape[0]
        half_batch_size = int(batch_size / 2)
        batch_num = int(sample_num / half_batch_size) + 1

        # index for minibatch training
        shuffled_idx = np.random.permutation(sample_num)

        # roop of batch
        for i_batch in range(batch_num):
            if half_batch_size*i_batch < sample_num:
                # ---------------------------
                # training of discriminator
                # ---------------------------

                # real data
                real_x = real_datas[shuffled_idx[half_batch_size*i_batch : half_batch_size*(i_batch+1)]]
                x_num = real_x.shape[0]
                real_y = np.ones((x_num, 1)) # label = 1
                
                # fake data
                latents = np.random.normal(0, 1, (x_num, self.latent_dim))
                fake_x = self.gene_model.predict(latents)
                fake_y = np.zeros((x_num, 1)) # label = 0
                
                #
                x = np.append(real_x, fake_x, axis=0)
                y = np.append(real_y, fake_y, axis=0)
                disc_loss = self.disc_model.train_on_batch(x=x, y=y)


                # ---------------------------
                # training of generator
                # ---------------------------

                # generated data
                # batch size = x_num * 2 (= real + fake in disc training)
                latents = np.random.normal(0, 1, (x_num * 2, self.latent_dim)) # x_num * 2
                y = np.ones((x_num * 2, 1)) # label = 1
                #
                gene_loss = self.combined_model.train_on_batch(x=latents, y=y)

                # training progress
                if print_on_batch:
                    print_epoch = now_epoch if now_epoch is not None else 0
                    print ("epoch: %d, batch: %d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (print_epoch, i_batch+1, disc_loss[0], 100*disc_loss[1], gene_loss))
                
        # training progress
        print_epoch = now_epoch if now_epoch is not None else 0
        print ("epoch: %d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (print_epoch, disc_loss[0], 100*disc_loss[1], gene_loss))
        #print ("epoch: %d, [D loss: %f, acc.: %.2f%%]" % (print_epoch, disc_loss[0], 100*disc_loss[1]))

        return

    def train_step_only_disc_with_random_noise(self, real_datas, batch_size=32, now_epoch=None, print_on_batch=False):
        '''
        training gan model on one epoch
        discriminatorのみ学習
        '''
        
        #
        sample_num = real_datas.shape[0]
        half_batch_size = int(batch_size / 2)
        batch_num = int(sample_num / half_batch_size) + 1

        # index for minibatch training
        shuffled_idx = np.random.permutation(sample_num)

        # roop of batch
        for i_batch in range(batch_num):
            if half_batch_size*i_batch < sample_num:
                # ---------------------------
                # training of discriminator
                # ---------------------------

                # real data
                real_x = real_datas[shuffled_idx[half_batch_size*i_batch : half_batch_size*(i_batch+1)]]
                x_num = real_x.shape[0]
                real_y = np.ones((x_num, 1)) # label = 1
                
                # fake data
                fake_x = np.random.rand(x_num, 2) * 2 - 1
                fake_y = np.zeros((x_num, 1)) # label = 0
                
                #
                x = np.append(real_x, fake_x, axis=0)
                y = np.append(real_y, fake_y, axis=0)
                disc_loss = self.disc_model.train_on_batch(x=x, y=y)
                
                # training progress
                if print_on_batch:
                    print_epoch = now_epoch if now_epoch is not None else 0
                    print ("epoch: %d, batch: %d, [D loss: %f, acc.: %.2f%%]" % (print_epoch, i_batch+1, disc_loss[0], 100*disc_loss[1]))
                
        # training progress
        print_epoch = now_epoch if now_epoch is not None else 0
        print ("epoch: %d, [D loss: %f, acc.: %.2f%%]" % (print_epoch, disc_loss[0], 100*disc_loss[1]))

        return

    def train_step_only_gene(self, real_datas, batch_size=32, now_epoch=None, print_on_batch=False):
        '''
        training gan model on one epoch
        generatorのみ学習
        '''
        
        #
        sample_num = real_datas.shape[0]
        half_batch_size = int(batch_size / 2)
        batch_num = int(sample_num / half_batch_size) + 1

        # index for minibatch training
        shuffled_idx = np.random.permutation(sample_num)

        # roop of batch
        for i_batch in range(batch_num):
            if half_batch_size*i_batch < sample_num:
                # real data
                real_x = real_datas[shuffled_idx[half_batch_size*i_batch : half_batch_size*(i_batch+1)]]
                x_num = real_x.shape[0]
                
                # ---------------------------
                # training of generator
                # ---------------------------

                # generated data
                # batch size = x_num * 2 (= real + fake in disc training)
                latents = np.random.normal(0, 1, (x_num * 2, self.latent_dim)) # x_num * 2
                y = np.ones((x_num * 2, 1)) # label = 1
                #
                gene_loss = self.combined_model.train_on_batch(x=latents, y=y)

                # training progress
                if print_on_batch:
                    print_epoch = now_epoch if now_epoch is not None else 0
                    #print ("epoch: %d, batch: %d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (print_epoch, i_batch+1, disc_loss[0], 100*disc_loss[1], gene_loss))
                    #print ("epoch: %d, batch: %d, [D loss: %f, acc.: %.2f%%]" % (print_epoch, i_batch+1, disc_loss[0], 100*disc_loss[1]))
                    print ("epoch: %d, batch: %d, [G loss: %f]" % (print_epoch, i_batch+1, gene_loss))
                
        # training progress
        print_epoch = now_epoch if now_epoch is not None else 0
        #print ("epoch: %d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (print_epoch, disc_loss[0], 100*disc_loss[1], gene_loss))
        #print ("epoch: %d, [D loss: %f, acc.: %.2f%%]" % (print_epoch, disc_loss[0], 100*disc_loss[1]))
        print ("epoch: %d, [G loss: %f]" % (print_epoch, gene_loss))

        return

