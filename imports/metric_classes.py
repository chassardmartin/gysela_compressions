from .math_tools import psnr, hsnr 


class psnrMetric:

    def __init__(self, tensor1, tensor2):

        self.tensor1 = tensor1 
        self.tensor2 = tensor2 

    def compute(self, dimensions=None):
        """
        input : - dimensions is a tuple of int/Nonetype to choose dimensions to
                    project tensors on
        example : dimensions = (None, 8 , None) -> x = x[:,8,:] 
        """
        if dimensions is None:
            _slice = (slice(None),) * len(self.tensor1.shape) 
        else:
            _slice = () 
            for keep in dimensions:
                if keep is None:
                    _slice += (slice(None),) 
                else:
                    _slice += (slice(keep, keep+1),)

        x = self.tensor1[_slice] 
        y = self.tensor2[_slice]
        return psnr(x, y) 


class hsnrMetric:

    def __init__(self, p, tensor1, tensor2):

        self.parameter = p 
        self.tensor1 = tensor1 
        self.tensor2 = tensor2 

    def compute(self, dimensions=None):
        """
        input : - dimensions is a tuple of int/Nonetype to choose dimensions to
                    project tensors on
        example : dimensions = (None, 8 , None) -> x = x[:,8,:] 
        """
        if dimensions is None:
            _slice = (slice(None),) * len(self.tensor1.shape) 
        else:
            _slice = () 
            for keep in dimensions:
                if keep is None:
                    _slice += (slice(None),) 
                else:
                    _slice += (slice(keep, keep+1),)
        x = self.tensor1[_slice] 
        y = self.tensor2[_slice] 
        return hsnr(x, y, self.parameter)
