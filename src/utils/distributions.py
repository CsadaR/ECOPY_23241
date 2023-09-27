import random
import math


class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand  # Véletlenszám generátor
        self.loc = loc  # x0 (location)
        self.scale = scale  # gamma (scale)

    def pdf(self, x):
        # Számítjuk ki a Cauchy-eloszlás valószínűségi sűrűségfüggvényét
        if self.scale <= 0:
            raise ValueError("A skála (scale) értéke pozitívnek kell lennie.")

        probability_density = 1 / (math.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2))
        return probability_density

    def cdf(self, x):
        # Számítjuk ki a Cauchy-eloszlás kumulatív eloszlásfüggvényét
        if self.scale <= 0:
            raise ValueError("A skála (scale) értéke pozitívnek kell lennie.")

        cumulative_probability = 0.5 + (1 / math.pi) * math.atan((x - self.loc) / self.scale)
        return cumulative_probability

    def gen_random(self):
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