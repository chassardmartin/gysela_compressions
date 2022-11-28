import pandas as pd


class IdentityDiagAnalysis:
    def __init__(self, diag_dir, compressor_list):

        self.diag_dir = diag_dir
        self.compressor_list = compressor_list
        self.results = {}
        self.results["compression_method"] = []
        self.results["metric_used"] = []
        self.results["quality_value"] = []

    def add_metric(self, metric_name, quality_list):

        for compressor, value in zip(self.compressor_list, quality_list):

            self.results["compression_method"].append(compressor.__name__)
            self.results["metric_used"].append(metric_name)
            self.results["quality_value"].append(value)

    def results_to_json(self):

        df = pd.DataFrame(self.results)
        df.to_json(self.diag_dir + "identity.json")


class FourierDiagAnalysis:
    def __init__(self, diag_dir, compressor_list):

        self.diag_dir = diag_dir
        self.compressor_list = compressor_list
        self.results = {}
        self.results["compression_method"] = []
        self.results["metric_used"] = []
        self.results["quality_value"] = []

    def add_metric(self, metric_name, quality_list):

        for compressor, value in zip(self.compressor_list, quality_list):

            self.results["compression_method"].append(compressor.__name__)
            self.results["metric_used"].append(metric_name)
            self.results["quality_value"].append(value)

    def results_to_json(self):

        df = pd.DataFrame(self.results)
        df.to_json(self.diag_dir + "fourier.json")


class GYSELAmostunstableDiagAnalysis:
    def __init__(self, diag_dir, compressor_list):

        self.diag_dir = diag_dir
        self.compressor_list = compressor_list
        self.results = {}
        self.results["compression_method"] = []
        self.results["metric_used"] = []
        self.results["quality_value"] = []

    def add_metric(self, metric_name, quality_list):

        for compressor, value in zip(self.compressor_list, quality_list):

            self.results["compression_method"].append(compressor.__name__)
            self.results["metric_used"].append(metric_name)
            self.results["quality_value"].append(value)

    def results_to_json(self):

        df = pd.DataFrame(self.results)
        df.to_json(self.diag_dir + "GYSELAmostunstable.json")
