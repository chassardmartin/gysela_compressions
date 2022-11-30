from .math_tools import psnr, hsnr
import dask.array as da


class psnrMetric:
    def __init__(self, tensor1, tensor2):
        """
        Initialize the metric with two tensors to compare 
        """
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.__name__ = "psnr"

    def compute(self, time_series):
        """
        computes the error in that metric
        -time_series : bool, False -> we consider the whole tensor 
                                as the data. Output is a scalar (one-element list)
                             True ->  the errors are computed time_wise.
                                        Output is a list. 
        
        """
        if not time_series:
            x = self.tensor1
            y = self.tensor2
            res = psnr(x, y)
            # If tensors were dask arrays
            if type(res) is da.core.Array:
                res = res.compute()
            # we return a one-element list to have type coherence
            return [res]
        else:
            res = []
            dim = len(self.tensor1.shape)
            T = self.tensor1.shape[0]
            for t in range(T):
                _slice = (slice(t, t + 1),)
                _slice += (slice(None),) * (dim - 1)
                x = self.tensor1[_slice]
                y = self.tensor2[_slice]
                current_res = psnr(x, y)
                # If tensors were dask arrays
                if type(current_res) is da.core.Array:
                    current_res = current_res.compute()
                res.append(current_res)
            return res


class hsnrMetric:
    def __init__(self, p, tensor1, tensor2):
        """
        Initialize the metric with two tensors to compare 
        - p : parameter in the definition of the metric, see hsnr
        """

        self.parameter = p
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.__name__ = "hsnr" + "_" + str(p)

    def compute(self, time_series):
        """
        computes the error in that metric
        -time_series : bool, False -> we consider the whole tensor 
                                as the data. Output is a scalar (one-element list)
                             True ->  the errors are computed time_wise.
                                        Output is a list. 
        """
        if not time_series:
            x = self.tensor1
            y = self.tensor2
            res = hsnr(self.parameter, x, y)
            if type(res) is da.core.Array:
                res = res.compute()
            # we return a one-element list to have type coherence 
            return [res]
        else:
            res = []
            dim = len(self.tensor1.shape)
            T = self.tensor1.shape[0]
            for t in range(T):
                _slice = (slice(t, t + 1),)
                _slice += (slice(None),) * (dim - 1)
                x = self.tensor1[_slice]
                y = self.tensor2[_slice]
                current_res = hsnr(self.parameter, x, y)
                if type(current_res) is da.core.Array:
                    current_res = current_res.compute()
                res.append(current_res)
            return res
