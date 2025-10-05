class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


# 实例
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # 苹果的总价
orange_price = mul_orange_layer.forward(orange, orange_num)  # 橘子的总价
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # 总价
price = mul_tax_layer.forward(all_price, tax)  # 含税总价

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # 含税总价的梯度
dapple_price, dorange_price = add_apple_orange_layer.backward(
    dall_price)  # 总价的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # 苹果的梯度
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # 橘子的梯度

# 输出
print(price)
print(dapple_num, dapple, dorange_num, dorange, dtax)
