from numpy import arange, empty_like, zeros, sum, isnan, any, random
from modules.filter.probability import g_pdf

def predict(hypothesis, odom, odom_variance):
    """
    @brief Kalman Filter prediction for Gaussian variable
    states and linear transition state estimation.

    @param hypothesis: Hypothesis.
    @param odom: float.
        The offset of the vehicle during motion computed using odometry w.r.t. its local frame.
    @param odom_variance: float.
        The variance of the odometry.
    """
    # The hypothesis assumed to be a gaussian-distributed pdf. 
    # Therefore sum the mean and the variance.
    hypothesis.mean += odom
    hypothesis.variance += odom_variance
    return

def update(hypothesis, P_Y_given_x_r, nsamples=10000):
    """
    @param hypothesis: Hypothesis.
        The hypothesis to be updated.
    @param P_Y_given_x_r: function.
        The probability distribution function of the measurement given the position x and the route of the hypothesis
    @param nsamples: int (default=10000).
        The number of sampled points for the filter.
    """

    # Information of the hypothesis
    mu_prime = hypothesis.mean
    var_prime = hypothesis.variance
    weight_prime = hypothesis.weight
    sigma_prime = var_prime ** 0.5

    # Discretization of the route position to the array ddomain
    route = hypothesis.route

    # The new (unnormalized) belief after update
    def tau(x):
        return P_Y_given_x_r(x, route) * g_pdf(x, mu_prime, sigma_prime)

    # As RBPF, sample around the mode points and evaluate their likelihood.
    sampled_points_x = random.normal(mu_prime, sigma_prime, 10000)
    mu_bar = 0
    weight_bar = 0
    tau_list = [tau(x) for x in sampled_points_x]
    eta = sum( tau_list )
    for x in sampled_points_x:
        tau_x = tau(x)
        mu_bar += x * tau_x 
        weight_bar += tau_x
    mu_bar *= 1./eta
    weight_bar = weight_bar *  weight_prime # New weight of the hypothesis
    var_bar = 0
    for x in sampled_points_x:
        var_bar += tau(x) * ((x - mu_bar) ** 2)
    var_bar *= 1./eta
    
    if( any( isnan([mu_bar, var_bar, weight_bar]) ) ):
        raise Exception("Error: nan value found. mu_bar: {} | var_bar : {} | weight_bar : {} ".format(mu_bar,var_bar,weight_bar))

    hypothesis.mean = mu_bar
    hypothesis.variance = var_bar
    hypothesis.weight = weight_bar
    return