# Content controllable motion infilling version 2 \ (+ latent space random sampling)

Motion infilling into target content:

The [Baseline](https://arxiv.org/abs/2010.11531)(Convolutional Autoencoders for Human Motion Infilling, 3DV 2020) can generate only one determined output from one input. But, there are many possible cases between keyframes. Therefore, this project conducted for making its output various with conditional input.

Additional: Baseline [Implementation(pytorch version)](https://github.com/rlgnswk/Motion-Infilling-pytorch-version-implementation) 

-----------------

## Overall Structure:
<p float="center">
  <img src="./figs/model_overview2.png" width="700" />

</p>


## Result (random sampling the feature space of Transition Encoder):
<p float="center">
  <img src="./figs/random_end1.gif" width="500" />
  <img src="./figs/random_mid1.gif" width="500" />
   
</p>


## Result (motion A - random transition - motion B):
<p float="center">
  <img src="./figs/random_AB.gif" width="500" />
  <img src="./figs/random_AB2.gif" width="500" />
   
</p>


