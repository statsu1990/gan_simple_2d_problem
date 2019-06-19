import gan

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 今更きけないGAN
# https://qiita.com/triwave33/items/1890ccc71fab6cbca87e

class GanTest2D:
    def __init__(self, data_num, latent_dim, train_epoch):
        self.DATA_NUM = data_num
        self.LATENT_DIM = latent_dim
        self.TRAIN_EPOCH = train_epoch

        self.real_datas = None
        
        return

    # run
    def run(self):
        # make real data
        self.make_real_data(self.DATA_NUM)
        self.__plot_scat1(self.real_datas[:,0], self.real_datas[:,1], label='real data')

        # make gan model
        self.make_gan_model()
        #self.make_gan_model_separating_disc_gene()
        
        # graph of real and judged as true by discriminator data
        self.__check_disc(self.gan, 100)
        
        return

    # data
    def make_real_data(self, data_num):
        self.real_datas = self.__sample_data_in_circle(data_num, radius=0.5)
        #self.real_datas = self.__sample_data_in_half_circle(data_num, radius=0.5)
        return
    def __sample_data_in_circle(self, data_num, radius):
        #
        center = np.array([0.5, 0.5])
        #center = np.array([0.0, 0.0])

        # sampling num
        sampling_margin = 2
        sampling_num = int((1.0 * 1.0) / (radius * radius * 3.14) * data_num * sampling_margin)
        
        # sampling
        end_sampling_flag = False
        x = np.empty((0,2), float)
        
        # sampling roop
        while not end_sampling_flag:
            # x in [-1,1)
            x_sampled = np.random.rand(sampling_num, 2) * 2.0 - 1.0
            x_sampled = x_sampled[np.sqrt(np.sum(np.square(x_sampled - center), axis=1)) <= radius]

            #
            x = np.append(x, x_sampled, axis=0)

            # check flag
            end_sampling_flag = x.shape[0] >= data_num

        # extract
        x = x[0:data_num]

        return x
    def __sample_data_in_half_circle(self, data_num, radius):
        #
        center = np.array([0.5, 0.5])
        #center = np.array([0.0, 0.0])

        # sampling num
        sampling_margin = 2
        sampling_num = int((1.0 * 1.0) / (radius * radius * 3.14) * data_num * sampling_margin)
        
        # sampling
        end_sampling_flag = False
        x = np.empty((0,2), float)
        
        # sampling roop
        while not end_sampling_flag:
            # x in [-1,1)
            x_sampled = np.random.rand(sampling_num, 2) * 2.0 - 1.0
            x_sampled = x_sampled[np.sqrt(np.sum(np.square(x_sampled - center), axis=1)) <= radius]
            x_sampled = x_sampled[x_sampled[:,1] < center[1]]

            #
            x = np.append(x, x_sampled, axis=0)

            # check flag
            end_sampling_flag = x.shape[0] >= data_num

        # extract
        x = x[0:data_num]

        return x


    # gan
    def make_gan_model(self):
        # make model
        self.gan = gan.GAN(latent_dim=self.LATENT_DIM, data_dim=self.real_datas.shape[1])
        #self.gan.make_model(gene_hidden_neurons=[32, 16, 16], disc_hidden_neurons=[32, 16, 16])
        self.gan.make_model(gene_hidden_neurons=[32, 16, 16], disc_hidden_neurons=[124, 64, 16])
        
        # train gan model
        fig = plt.figure()
        ims = []
        ims.append([self.__plot_gene_data(self.gan, data_num=3000, show=False)])
        # training epoch roop
        for iep in range(self.TRAIN_EPOCH):
            self.gan.train_step(self.real_datas, batch_size=32, now_epoch=iep)
            #self.gan.train_step_test1(self.real_datas, batch_size=32, now_epoch=iep)
            
            # images for animation
            ims.append([self.__plot_gene_data(self.gan, data_num=3000, show=False)])

        # graph of real and generated data
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save('generated_point.gif', writer='pillow')
        plt.show()

        return
    def make_gan_model_separating_disc_gene(self):
        # make model
        self.gan = gan.GAN(latent_dim=self.LATENT_DIM, data_dim=self.real_datas.shape[1])
        #self.gan.make_model(gene_hidden_neurons=[32, 16, 16], disc_hidden_neurons=[32, 16, 16])
        self.gan.make_model(gene_hidden_neurons=[32, 16, 16], disc_hidden_neurons=[248, 124, 16])
        
        # train disc model
        for iep in range(self.TRAIN_EPOCH):
            self.gan.train_step_only_disc_with_random_noise(self.real_datas, batch_size=32, now_epoch=iep)
        
        # train gene model
        fig = plt.figure()
        ims = []
        ims.append([self.__plot_gene_data(self.gan, data_num=3000, show=False)])
        # training epoch roop
        for iep in range(self.TRAIN_EPOCH):
            self.gan.train_step_only_gene(self.real_datas, batch_size=32, now_epoch=iep)
            
            # images for animation
            ims.append([self.__plot_gene_data(self.gan, data_num=3000, show=False)])

        # graph of real and generated data
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save('generated_point.gif', writer='pillow')
        plt.show()

        return

    def __plot_gene_data(self, gan, data_num, title=None, show=True):
        '''
        plot generated data
        '''
        latents = np.random.normal(0, 1, (300, self.LATENT_DIM))
        gene_datas = gan.gene_model.predict(latents)
        image = self.__plot_scat1(gene_datas[:,0], gene_datas[:,1], color='c', title=title, show=show)
        return image
    def __plot_disc_predict(self, gan, data_num, binary=False, save=False, savefilename=''):
        '''
        plot discrimination model prediction
        '''
        # grid [-1,1] and [-1,1]
        x1d = np.linspace(start=-1, stop=1, num=data_num)
        x1, x2 = np.meshgrid(x1d, x1d)
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)
        #
        x = np.concatenate([x1[:,np.newaxis], x2[:,np.newaxis]], axis=1)

        # discriminate model prediction
        if binary:
            pre = (self.gan.disc_model.predict(x) > 0.5) * 1
        else:
            pre = gan.disc_model.predict(x)

        #
        self.__plot_scat1(x[:,0], x[:,1], color=np.ravel(pre), label='disc predict', save=save, savefilename=savefilename)


        return
    def __check_disc(self, gan, data_num):
        real_disc = self.gan.disc_model.predict(self.real_datas)
        real_disc_acc = np.average(real_disc > 0.5)
        real_disc_ave = np.average(real_disc)
        real_disc_std = np.std(real_disc)
        print('real disc acc {0}, ave {1}, std {2}'.format(real_disc_acc, real_disc_ave, real_disc_std))

        self.__plot_disc_predict(self.gan, 100, binary=False, save=False)
        self.__plot_disc_predict(self.gan, 100, binary=True, save=True, savefilename='discriminate_true_range.png')
        
        return

    # plot
    def __plot_scat1(self, x, y, color=None, title=None, label=None, xmin=-1, xmax=1, ymin=-1, ymax=1, show=True, save=False, savefilename=''):
        image = plt.scatter(x, y, c=color, s=5, label=label, cmap='Blues')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(title)
        if (color is not None) and (type(color) != type('string')):
            plt.colorbar()
        if save:
            plt.savefig(savefilename)
        if show:
            plt.show()
        
        return image


if __name__ == '__main__':
    gan_test_2d = GanTest2D(500, 16, 200)
    gan_test_2d.run()
