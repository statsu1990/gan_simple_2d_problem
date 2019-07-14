# gan_simple_2d_problem
Applied gan and wgan-gp to a simple two-dimensional problem to understand the features of gan and wgan-gp.
This program was implemented with using python and keras.

詳細は以下のブログ参照。<br>
https://st1990.hatenablog.com/entry/2019/06/20/010919

The details are described in the following blog.<br>
https://st1990.hatenablog.com/entry/2019/06/20/010919


## simple 2d problem
Create a model that generates the following real data with gan.
![mrc](https://github.com/statsu1990/gan_simple_2d_problem/blob/master/images/real_data_dist.png)<br>

## result
The progress of learning of the wgangp generating model is following.
![mrc](https://github.com/statsu1990/gan_simple_2d_problem/blob/master/images/generated_point_wgangp.gif)<br>

The progress of learning of the gan generating model is following.
![mrc](https://github.com/statsu1990/gan_simple_2d_problem/blob/master/images/generated_point.gif)<br>

The areas that the gan disciriminating model recognizes data as real is following.
![mrc](https://github.com/statsu1990/gan_simple_2d_problem/blob/master/images/discriminate_true_range.png)<br>

