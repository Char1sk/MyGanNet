class LossRecord():
    def __init__(self):
        self.D = 0.0
        self.G = 0.0
        self.GAdv = 0.0
        self.GPxl = 0.0
        self.GPer = 0.0
        self.counter = 0
    
    def add(self, D:float, G:float, GAdv:float, GPxl:float, GPer:float) -> None:
        self.D += D
        self.G += G
        self.GAdv += GAdv
        self.GPxl += GPxl
        self.GPer += GPer
    
    def div(self, times:int) -> None:
        self.D /= times
        self.G /= times
        self.GAdv /= times
        self.GPxl /= times
        self.GPer /= times
    
    def clear(self) -> None:
        self.D = 0.0
        self.G = 0.0
        self.GAdv = 0.0
        self.GPxl = 0.0
        self.GPer = 0.0
