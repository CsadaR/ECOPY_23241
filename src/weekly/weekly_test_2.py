
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

    def gen_rand(self):
        import math
        import random
        u = random.uniform(0, 1)
        if u < 0.5:
            return self.loc + self.scale * math.log(2 * u)
        else:
            return self.loc - self.scale * math.log(2 - 2 * u)

    def mean(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        else:
            return self.loc

    def variance(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        else:
            return 2 * (self.scale ** 2)

    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        else:
            return 0.0

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        else:
            return 3.0

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        else:
            return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


class ParetoDistribution:
    def __init__(self, random, scale, shape):
        self.random = random
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x >= self.scale:
            return (self.shape * (self.scale ** self.shape)) / (x ** (self.shape + 1))
        else:
            return 0.0

    def cdf(self, x):
        if x >= self.scale:
            return 1.0 - (self.scale / x) ** self.shape
        else:
            return 0.0

    def ppf(self, p):
        if 0 < p <= 1:
            return self.scale / ((1 - p) ** (1 / self.shape))
        else:
            raise ValueError("0 és 1 között kell lennie")

    def gen_rand(self):
        import random
        u = random.uniform(0, 1)
        return self.ppf(u)

    def mean(self):
        import math
        if self.shape <= 1:
            return math.inf
        else:
            return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        import math
        if self.shape <= 2:
            return math.inf
        else:
            return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        import math
        if self.shape <= 3:
            return math.inf
        else:
            return (2 * (1 + self.shape)) / (self.shape - 3) * math.sqrt((self.shape - 2) / self.shape)

    def ex_kurtosis(self):
        import math
        if self.shape <= 4:
            return math.inf
        else:
            return 6 * ((self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2) / (
                        self.shape * (self.shape - 3) * (self.shape - 4)))

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]
