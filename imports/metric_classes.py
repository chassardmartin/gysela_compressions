from .math_tools import psnr, hsnr


class psnrMetric:
    def __init__(self, tensor1, tensor2):

        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def compute(self, time_series=False):
        """
        computes the error in that metric
        -time_series : bool, True -> we consider the whole tensor 
                                as the data
                             Otherwise, the errors are computed time_wise
        """
        if time_series:
            x = self.tensor1
            y = self.tensor2
            return psnr(x, y)
        else:
            res = []
            T = self.tensor1.shape[0]
            for t in self.range(T):
                _slice = (slice(t, t + 1),)
                _slice += (slice(None),) * (T - 1)
                x = self.tensor1[_slice]
                y = self.tensor2[_slice]
                res.append(psnr(x, y))
            return res


class hsnrMetric:
    def __init__(self, p, tensor1, tensor2):

        self.parameter = p
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def compute(self, time_series=False):
        """
        computes the error in that metric
        -time_series : bool, True -> we consider the whole tensor 
                                as the data
                             Otherwise, the errors are computed time_wise
        """
        if time_series:
            x = self.tensor1
            y = self.tensor2
            return hsnr(self.parameter, x, y)
        else:
            res = []
            T = self.tensor1.shape[0]
            for t in self.range(T):
                _slice = (slice(t, t + 1),)
                _slice += (slice(None),) * (T - 1)
                x = self.tensor1[_slice]
                y = self.tensor2[_slice]
                res.append(hsnr(self.parameter, x, y))
            return res
