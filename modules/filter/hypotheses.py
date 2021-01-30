from .hypothesis import Hypothesis

class Hypotheses:

    def __init__(self):
        self.__dict__ = {}
        return

    def get(self,idx = None):
        if(idx is None):
            return self.__dict__
        return self.__dict__[idx]

    def normalize_weights(self):
        sum_w = 0
        for idx in self.__dict__.keys():
            sum_w += self.__dict__[idx].weight
        
        for idx in self.__dict__.keys():
            self.__dict__[idx].weight /= sum_w
        return

    def get_best_hypothesis(self):
        max_w_hypothesis = None
        hypotheses = self.__dict__
        for hypothesis in hypotheses.values():
            if max_w_hypothesis is None:
                max_w_hypothesis = hypothesis
                continue
            if max_w_hypothesis.weight < hypothesis.weight:
                max_w_hypothesis = hypothesis
        return max_w_hypothesis

    def create_hypothesis(self, mean, variance, route,weight):
        
        h = Hypothesis(
            mean = mean, 
            variance=variance, 
            weight=weight,
            route=route
        ) 
        self.__dict__[h.route.idx] = h
        w = 1. / len(self.__dict__)

        # Since hypotheses were the same up to that point, all of them have the same weight
        for idx in self.__dict__.keys():
            self.__dict__[idx].weight = w
        return 

    def prune_hypotheses(self, threshold=0.1):
        hypotheses = self.__dict__
        del_keys = []
        max_w = -1e10
        for idx in hypotheses.keys():
            if(max_w < hypotheses[idx].weight):
                max_w = hypotheses[idx].weight
        for idx in hypotheses.keys():
            if(hypotheses[idx].weight/max_w < threshold):
                del_keys.append(idx)
        for idx in del_keys:
            del hypotheses[idx]
        return del_keys