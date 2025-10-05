class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Goodbye " + self.name + "!")


m = Man("David")
m.hello()
m.goodbye()
