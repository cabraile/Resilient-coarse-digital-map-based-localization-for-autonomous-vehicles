"""
The probability density functions used for update

"""
def p_speedlimit_given_position_and_route(y, x, r,sensitivity, fpr):
    """
    The probability density function of measuring the segment feature 
    "speedlimit" given the position 'x' in the route 'r' (also given).

    @param y: int. 
        The speed limit detected.
    @param x: float.
        The position in the route.
    @param r: Route.
        The route from which the PDF is computed.
    @param sensitivity: float.
        The sensitivity rate of the detection module in [0.0,1.0]
    @param fpr: float
        The false positive rate of the detection module in [0.0, 1.0]
    @return float. The likelihood of the distribution in y.
    """
    way, _ = r.get_by_distance(x)
    norm = 1. /(fpr + sensitivity)
    if(way is None):
        return 0
    if y == way.maxspeed():
        prob = sensitivity * norm
    else:
        prob = fpr * norm
    return prob

def p_landmark_given_position_and_route(y, x, r, sensitivity, fpr):
    """
    The probability of measuring a landmark y given the distance x at a route r.

    TODO: In Naseer, Suger et al 2017, the returned probability is
    the similarity between the descriptor of the given image and the
    landmark image. Check later

    @param y: Landmark.
        The detected landmark.
    @param x: float.
        The position in the route.
    @param r: Route.
        The route from which the PDF is computed.
    @param sensitivity: float.
        The sensitivity rate of the detection module in [0.0,1.0]
    @param fpr: float
        The false positive rate of the detection module in [0.0, 1.0]

    @return float. The likelihood of the distribution in y.
    """
    lm_range = r.get_landmark_range(y.id)
    
    # Landmark should not exist in that route. False Positive detected
    norm = 1. /(fpr + sensitivity)
    if( lm_range is None):
        prob = fpr*norm
        return prob

    # When in range bounds, multiply by the sensitivity
    if (x >= lm_range[0]) and (x <= lm_range[1]):
        prob = sensitivity  * norm
    # Otherwise, multiply by the fpr
    else:
        prob = fpr * norm
    return prob
