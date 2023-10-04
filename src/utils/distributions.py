class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand  # Véletlenszám generátor
        self.loc = loc  # x0 (location)
        self.scale = scale  # gamma (scale)

    def pdf(self, x):
        # Számítjuk ki a Cauchy-eloszlás valószínűségi sűrűségfüggvényét
        import math
        if self.scale <= 0:
            raise ValueError("A skála (scale) értéke pozitívnek kell lennie.")

        probability_density = 1 / (math.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2))
        return probability_density

    def cdf(self, x):
        import math
        # Számítjuk ki a Cauchy-eloszlás kumulatív eloszlásfüggvényét
        if self.scale <= 0:
            raise ValueError("A skála (scale) értéke pozitívnek kell lennie.")

        cumulative_probability = 0.5 + (1 / math.pi) * math.atan((x - self.loc) / self.scale)
        return cumulative_probability

    def gen_random(self):
        import math
        # Generálunk egy Cauchy-eloszlású véletlen számot
        u = self.rand.random()  # Véletlenszerű szám az (0, 1) intervallumból
        random_number = self.loc + self.scale * math.tan(math.pi * (u - 0.5))
        return random_number

    def mean(self):
        # Számítjuk ki az eloszlásfüggvény átlagát
        raise Exception("Moment undefined")

    def median(self):
        # Az eloszlásfüggvény mediánja a lokáció értéke
        return self.loc


class LogisticDistribution:
        def __init__(self, rand, location, scale):
            self.rand = rand
            self.location = location
            self.scale = scale
        def generate_random(self):
            import math
            u = self.rand.random()  # Véletlenszerű szám az [0, 1) tartományból
            x = self.location + self.scale * math.log(u / (1 - u))  # Logistic eloszlás generálása
            return x
        def pdf(self, x):
            import math
            exponent = -(x - self.location) / self.scale
            pdf_value = (1 / (self.scale * (1 + math.exp(exponent)) ** 2)) * math.exp(exponent)
            return pdf_value
        def cdf(self, x):
            import math
            z = (x - self.location) / self.scale
            cdf_value = 1 / (1 + math.exp(-z))
            return cdf_value
        def ppf(self, p):
            import math
            if p < 0 or p > 1:
                raise ValueError("A valószínűségi értéknek az [0, 1] tartományban kell lennie.")
            if p == 0:
                return float("-inf")
            if p == 1:
                return float("inf")
            ppf_value = self.location + self.scale * math.log(p / (1 - p))
            return ppf_value
        def gen_rand(self):
            return self.generate_random()
        def mean(self):
            if self.scale == 0:
                raise Exception("Moment undefined")
            return self.location
        def variance(self):
            import math
            if self.scale == 0:
                raise Exception("Moment undefined")
            variance = (math.pi ** 2) * (self.scale ** 2) / 3
            return variance
        def skewness(self):
            if self.scale == 0:
                raise Exception("Moment undefined")
            skewness = 0
            return skewness
        def ex_kurtosis(self):
            if self.scale == 0:
                raise Exception("Moment undefined")
            ex_kurtosis = 1.2
            return ex_kurtosis
        def mvsk(self):
            if self.scale == 0:
                raise Exception("Moment undefined")
            return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof
    def pdf(self, x):
        import math
        if x < 0:
            return 0
        coefficient = (1 / (2**(self.dof / 2))) * (1 / math.gamma(self.dof / 2))
        exponent = (-x / 2)
        return coefficient * (x**(self.dof / 2 - 1)) * math.exp(exponent)
    def cdf(self, x):
        import scipy.special
        if x < 0:
            return 0
        return scipy.special.gammainc(self.dof / 2, x / 2)
    def ppf(self, p):
        import scipy.special
        if p < 0 or p > 1:
            raise ValueError("Moment undefined")
        return 2 * scipy.special.gammaincinv(self.dof / 2, p)

    def gen_rand(self):
        import random
        u = random.uniform(0, 1)
        return self.ppf(u)
    def mean(self):
        try: return self.dof
        except: raise ValueError("Moment undefined")
    def variance(self):
        try: return 2 * self.dof
        except:
            raise ValueError("Moment undefined")
    def skewness(self):
        import math
        try: return math.sqrt(8 / self.dof)
        except:
            raise ValueError("Moment undefined")
    def ex_kurtosis(self):
        try: return 12 / self.dof
        except:
            raise ValueError("Moment undefined")
    def mvsk(self):
            mean = self.mean()
            variance = self.variance()
            skewness = self.skewness()
            ex_kurtosis = self.ex_kurtosis()
            try: return [mean, variance, skewness, ex_kurtosis]
            except:
                raise ValueError("Moment undefined")