import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.isInitialized = False
        self.opt = None
    
    def initialize(self) -> None:
        pass
    
    def parse(self) -> argparse.Namespace:
        if not self.isInitialized:
            self.initialize()
            self.isInitialized = True
        
        self.opt = self.parser.parse_args()
        
        return self.opt