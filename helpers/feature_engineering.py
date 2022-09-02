import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


class FeatureEngineering:
    def __init__(self, dataset: pd.DataFrame()):
        self.dataset = dataset

    def transform_dataset(self):
        self.dataset["periodOfDay"] = self.dataset["hour"].apply(lambda x: x % 8)

        amount_third_quantile = self.dataset["amount"].quantile(0.75)

        self.dataset["higherValue"] = self.dataset["amount"].apply(
            lambda x: 1 if x >= amount_third_quantile else 0
        )

        oldbalance_third_quantile = self.dataset["oldbalanceOrg"].quantile(0.75)

        self.dataset["higherOldBalance"] = self.dataset["oldbalanceOrg"].apply(
            lambda x: 1 if x >= oldbalance_third_quantile else 0
        )

        conditions = [
            (self.dataset["amount"] == self.dataset["oldbalanceOrg"]),
            (self.dataset["amount"] != self.dataset["oldbalanceOrg"]),
        ]

        values = [1, 0]

        self.dataset["withdrawAll"] = np.select(conditions, values)

        self.dataset["amountSquare"] = np.power(self.dataset["amount"], 2)
        self.dataset["amountDiff"] = (
            np.mean(self.dataset["amount"]) - self.dataset["amount"]
        )

        return self.dataset
