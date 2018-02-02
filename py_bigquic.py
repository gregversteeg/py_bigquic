"""
A python front-end for BigQuic, a large-scale GLASSO solver written in R
Original paper and R code: http://bigdata.ices.utexas.edu/software/1035/

This script by:
Greg Ver Steeg <gregv@isi.edu>, 2018
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.covariance import log_likelihood
try:
    import rpy2
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri as swap

    R = rpy2.robjects.r
    importr('BigQuic')
except:
    msg = "Either R, rpy2, or BigQuic are not installed"
    raise ImportError(msg)


def bigquic(data, alpha=1.):
    """Python front-end for BigQuic GLASSO solver."""
    # Normalize data, save STDs
    data = np.copy(data)
    data -= data.mean(axis=0)
    stds = np.std(data, axis=0).clip(1e-10)
    data /= stds
    rdata = swap.numpy2ri(data)  # Load data into R space

    # Construct and run the program in R
    program_string = 'f <- function(r) {out = BigQuic(X=r, lambda=%f, use_ram=TRUE); ' % alpha
    program_string += 'as(out$precision_matrices[[1]], "matrix")}'
    f = R(program_string)
    prec = np.array(f(rdata))
    prec = 1. / stds * prec / stds[:, np.newaxis]  # Put back the original scaling
    return prec

def bigquic_cv(x, alphas=[0.2, 0.4, 0.8, 1.6], k=3, verbose=True):
    """Choose sparsity hyper-parameter using k-fold cross validation, searching
    a pre-defined grid of hyper-parameters.
    The documentation recommends NOT to use alpha < 0.4 for performance reasons..."""
    kf = KFold(n_splits=k)
    cv_dict = {}
    for alpha in alphas:
        for x_train, x_test in kf.split(x):
            try:
                prec = bigquic(x[x_train], alpha)
                emp_cov = np.cov(x[x_test].T)
                score = log_likelihood(emp_cov, prec)
                cv_dict[alpha] = cv_dict.get(alpha, []) + [score]
            except:
                pass
    best_alpha = sorted([(-np.mean(v), k) for k, v in cv_dict.iteritems()])[0][1]
    if verbose:
        print cv_dict
        print("best alpha: {}".format(best_alpha))
    return bigquic(x, best_alpha)

if __name__ == '__main__':  # Compare functionality to sklearn
    # Test adapted from http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html
    from sklearn.datasets import make_sparse_spd_matrix
    from sklearn.covariance import GraphLasso, log_likelihood
    from scipy.linalg import pinvh
    import matplotlib.pyplot as plt

    n_samples = 60
    n_features = 20
    alpha = 0.25  # Sparsity hyper parameter
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(n_features, alpha=.98,
                                  smallest_coef=.4,
                                  largest_coef=.7,
                                  random_state=prng)
    cov = pinvh(prec)
    # Normalize
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    # Generate data
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    # Fit different methods
    model = GraphLasso(alpha=alpha)
    model.fit(X)
    cov_ = model.covariance_
    prec_ = model.precision_

    emp_cov = np.cov(X.T)
    emp_prec = pinvh(emp_cov)

    bq_prec = bigquic(X, alpha)
    bq_cov = pinvh(bq_prec)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)

    # plot the covariances
    covs = [('Empirical', emp_cov), ('BigQUIC', bq_cov),
            ('GraphLasso', cov_), ('True', cov)]
    vmax = cov_.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 4, i + 1)
        plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s covariance' % name)


    # plot the precisions
    precs = [('Empirical', pinvh(emp_cov)), ('BigQUIC', bq_prec),
             ('GraphLasso', prec_), ('True', prec)]
    vmax = .9 * prec_.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 4, i + 5)
        plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s precision' % name)
        ax.set_axis_bgcolor('.7')

    plt.savefig('glasso_comparison.png')
    print("Interesting... The documentation for sklearn GLASSO says it doesn't penalize the L1 on the "
          "diagonal. The precision looks different, but it doesn't seem to have a big effect on LL.")
    print("Log likelihood for sklearn GLASSO: {} vs BIGQUIC GLASSO: {}".format(
            log_likelihood(emp_cov, prec_), log_likelihood(emp_cov, bq_prec)))

    print("Test cross-validation to choose hyper-parameter")
    bigquic_cv(X)