# Probabilistic Principal Component Analysis

https://www.cs.columbia.edu/~blei/seminar/2020-representation/readings/TippingBishop1999.pdf

# Mixtures of Probabilistic Principal Component Analysers

https://miketipping.com/papers/met-mppca.pdf

Gaussian Mixture Model (GMM) :
- $\lambda = \{p_i, \mu_i, \Sigma_i\}_{i=1}^{M}$
- $p(x|\lambda) = \sum_{i=1}^{M} N(x|\mu_i, \Sigma_i)p_i$

Probabilistic PCA (PPCA) :
- $t \in \mathbb{R}^d$ observed, $x \in \mathbb{R}^q$ latent, $q < d$
- We assume that a relationship $t = Wx + \mu + \epsilon$ exists, with $\epsilon \sim N(0, \sigma^2I)$
- The likelihood of $t$ given $x$ is $p(t|x) = N(t|Wx + \mu, \sigma^2I)$
- We assume a prior $p(x) = N(x|0, I)$ and get :
- $p(t) = N(t|\mu, \Sigma)$, $\Sigma = WW^T + \sigma^2I$
- $p(x|t) = N(x|M^{-1}W^T(t - \mu), \sigma^2M^{-1})$, $M = W^TW + \sigma^2I$

Mixture of PPCA (MPPCA) : $p(x|\lambda) = \sum_{i=1}^{M} N(x|W_i z + \mu_i, \sigma^2I)p(z=i)$

Expériences :

- sur images (ou feature vectors) : mu_i ~ classe, Sigma_i ~ variation intra-classe

- Feature visualization : Soit f un vecteur de feature de taille c x h x w. En faisant une MPPCA sur f, on obtient quelques vecteurs de feature pour chaque "cluster" : correspondent-ils à des motifs particuliers ? Qu'est-ce qu'il se passe si on prend un autoencodeur, et qu'on boost la PCA juste d'un seul cluster sans toucher au reste ? Si on identifie des especes de steering vector, on devrait voir les features importantes boostées ?

# Project plan :

Experiments :
- Implement PPCA and MPPCA (with EM algorithm)
- Tests :
    - Computational complexity : PCA eigenvalue decomposition of covariance matrix vs EM algorithm of PPCA vs analytical MLE of PPCA
    - Compression : compare PCA per sample (~ PCA on columns of one image), PCA, PPCA, MPPCA
        - fidelity : reconstruction error (MSE, PSNR, SSIM)
            - PPCA does not minimize MSE if $\sigma^2 \neq 0$ but the optimal reconstruction is given by $...$.
        - compression ratio
    - Missing data :
        - PPCA can handle missing data (NaN in X) but PCA cannot as it needs the full matrix to do linear algebra, without any probabilistic model to handle missing data.
        - Missing data are naturally handled in the EM algorithm of PPCA by considering missing data as latent variables.
        - Baselines :
            - replace missing data by mean
            - compute covariance by the kind of online algorithm and update only the non-missing data rows and columns, then normalise each row and column by the number of samples that were used to compute the covariance. Using this covariance, do regular PCA.
    - Reconstruction of missing data
    - Denoising : noise is captured by $\sigma^2$ in PPCA.
        - PCA doesn't work that well as, even if low eigenvalues can be considered noise, some noise can also be captured by high eigenvalues especially if the noise is full rank. Explicitly modeling the noise in PPCA allows to separate the noise from the signal. Add noise to ground truth data, then compare how well PCA, PPCA and MPPCA can denoise it.
    - Outliers detection
        - likelihood of a sample given the model, again not possible with PCA as it is not probabilistic
    - Generation : generate samples from PPCA and MPPCA : latent ~ N(0, I) -> sample = f(latent)
        - Being a probabilistic model, PPCA can generate new samples by sampling from the latent space and then applying the generative model.
    - Clustering / classification : say that we do not extend on this, as it merely comes from the fact that MPPCA is akin to a GMM, and we already saw how this worked in class.
    - Feature visualization
- Test on synthetic data
- Test on MNIST
- Test on CIFAR-10



  - missing data
  - outliers
  - noise
  - generative model
  - mixture models