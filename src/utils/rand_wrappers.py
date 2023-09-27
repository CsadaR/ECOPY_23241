
def random_from_list(input_list):
    import random
    if isinstance(input_list, list) and len(input_list) > 0:
        return random.choice(input_list)
    else:
        raise ValueError("A bemenetnek egy nem üres lista kell lennie.")

def random_sublist_from_list(input_list,number_of_elements):
        import random
        if isinstance(input_list, list) and isinstance(number_of_elements, int):
            if len(input_list) > 0:
                if number_of_elements <= len(input_list):
                    return random.sample(input_list, number_of_elements)
                else:
                    raise ValueError("A kiválasztandó elemek száma nem lehet nagyobb, mint a bemeneti lista hossza.")
            else:
                raise ValueError("A bemeneti lista nem lehet üres.")
        else:
            raise TypeError("A bemenetnek egy lista és egy egész szám kell lennie.")



def hundred_large_random():
    import random
    output_list = []
    # 100-szor ismételjük a következőt
    for i in range(100):
        # generálunk egy véletlen számot 10 és 1000 között
        x = random.randint(10, 1000)
        # hozzáadjuk a számot a listához
        output_list.append(x)
    # visszaadjuk a listát
    return output_list

class UniformDistribution:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, x):
        if x < self.a or x > self.b:
            return 0.0
        else:
            return 1.0 / (self.b - self.a)

a = 1.0
b = 5.0
uniform_dist = UniformDistribution(a, b)

x_values = [0.0, 2.0, 3.0, 6.0]
for x in x_values:
    pdf_value = uniform_dist.pdf(x)
    print(f'pdf({x}) = {pdf_value}')
