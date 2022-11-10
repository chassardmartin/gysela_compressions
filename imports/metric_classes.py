from .math_tools import psnr, hsnr 


class psnrMetric:

    def __init__(self, tensor1, tensor2):

        self.tensor1 = tensor1 
        self.tensor2 = tensor2 

    def compute(self):
        return psnr(self.tensor1, self.tensor2) 


class hsnrMetric:

    def __init__(self, p, tensor1, tensor2):

        self.parameter = p 
        self.tensor1 = tensor1 
        self.tensor2 = tensor2 

    def compute(self):
        return hsnr(self.tensor1, self.tensor2, self.parameter)
