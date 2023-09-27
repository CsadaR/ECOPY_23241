
class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale
    def pdf(self, x):
        import math
        abs_diff = abs(x - self.loc)
        return (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)
    def cdf(self, x):
        import math
        if x < self.loc:
            return 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            return 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)
    def ppf(self, p):
        import math
        if p < 0 or p > 1:
            raise ValueError("0 és 1 között kell lennie")

        if p < 0.5:
            return self.loc + self.scale * math.log(2 * p)
        else:
            return self.loc - self.scale * math.log(2 * (1 - p))

    def gen_random(self):
        import math
        import random
        u = self.rand.random()
        if u < 0.5:
            return float(self.loc + self.scale * math.log(2 * u))
        else:
            return float(self.loc - self.scale * math.log(2 - 2 * u))

    def mean(self):
        return self.loc
        raise Exception("Moment undefined")
    def variance(self):
        return 2 * (self.scale ** 2)
        raise Exception("Moment undefined")
    def skewness(self):
        return 0.0
        raise Exception("Moment undefined")
    def ex_kurtosis(self):
        return 3.0
        raise Exception("Moment undefined")
    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]
        raise Exception("Moment undefined")




class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        import random
        if x >= self.scale:
            return (self.shape * (self.scale ** self.shape)) / (x ** (self.shape + 1))
        else:
            return 0.0

    def cdf(self, x):
        import random
        if x >= self.scale:
            return 1.0 - (self.scale / x) ** self.shape
        else:
            return 0.0

    def ppf(self, p):
        import random
        if 0 < p <= 1:
            return self.scale / ((1 - p) ** (1 / self.shape))
        else:
            raise ValueError("0 és 1 között kell lennie")

    def gen_random(self):
        import math
        import random
        u = self.rand.random()
        if u < 0.5:
            return float(self.scale * math.log(2 * u))
        else:
            return float(-self.scale * math.log(2 - 2 * u))

    def mean(self):
        return (self.shape * self.scale) / (self.shape - 1)
        raise Exception("Moment undefined")
    def variance(self):
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))
        raise Exception("Moment undefined")
    def skewness(self):
        import math
        return (2 * (1 + self.shape)) / (self.shape - 3) * math.sqrt((self.shape - 2) / self.shape)
        raise Exception("Moment undefined")
    def ex_kurtosis(self):
        return (6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2)) / (
                self.shape * (self.shape - 3) * (self.shape - 4))
        raise Exception("Moment undefined")
    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]
        raise Exception("Moment undefined")