from numpy import sqrt, exp, pi

def g_pdf(x, mu, sigma):
    """
    The gaussian PDF.

    @param x: float or ndarray of float. The x from which the likelihood of the PDF is computed.
    @param mu: float. The mean of the Gaussian distribution.
    @param sigma: float. The standard deviation of the Gaussian distribution.
    """
    mult_factor = 1./(sigma * sqrt(2 * pi))
    exp_factor = -((x - mu) ** 2. ) / (2. * (sigma ** 2))
    return mult_factor * exp(exp_factor)