class GLoss():
    def __init__(self):
        self.G = 0.0
        self.GAdv = 0.0
        self.GPxl = 0.0
        self.GPer = 0.0
    
    def add(self, GAdv:float, GPxl:float, GPer:float) -> None:
        self.G += GAdv + GPxl + GPer
        self.GAdv += GAdv
        self.GPxl += GPxl
        self.GPer += GPer
    
    def div(self, times:int) -> None:
        self.G /= times
        self.GAdv /= times
        self.GPxl /= times
        self.GPer /= times
    
    def clear(self) -> None:
        self.G = 0.0
        self.GAdv = 0.0
        self.GPxl = 0.0
        self.GPer = 0.0
    

class LossRecord():
    def __init__(self):
        self.D = 0.0
        self.Gtotal = GLoss()
        self.Gparts = [GLoss(), GLoss(), GLoss(), GLoss()]
        # G_global, G_tl, G_tr, G_d
    
    def divall(self, times:int) -> None:
        self.D /= times
        self.Gtotal.div(times)
        for g in self.Gparts:
            g.div(times)
    
