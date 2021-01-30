class Hypothesis():

    def __init__(self, mean, variance, weight, route):
        self.mean = mean
        self.variance = variance
        self.route = route
        self.weight = weight
        return

    def __str__(self):
        return "Hypothesis of mean = {} variance = {}".format(self.mean, self.variance)