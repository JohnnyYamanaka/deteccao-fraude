import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

class EvaluateModel():
    def __init__(self, model, dataset: pd.DataFrame()):
        self.model = model
        self.dataset = dataset

    def run_model(self, seed):
        X = self.dataset.drop('isFraud', axis=1).copy()
        y = self.dataset['isFraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            stratify=y, random_state=seed)

        fited_model = self.model.fit(X_train, y_train)
        pred = fited_model.predict(X_test)

        auc = roc_auc_score(y_test, pred)

        print(f'Auc score: {auc: .4f}')
        print('\nClassification Report')
        print(classification_report(y_test, pred, zero_division=0))


        df_pred = pd.DataFrame({'isFraud': y_test})
        df_evaluate = X_test.join(df_pred)
        df_evaluate['predicted'] = pred

        df_evaluate['financialLost'] = df_evaluate.apply(lambda x: -x['amount'] if 
            ((x['isFraud'] == 1) and (x['predicted'] == 0)) or
            ((x['isFraud'] == 0) and (x['predicted'] == 1))
            else 0, axis=1)

        print(f'Total of loss: ${df_evaluate["financialLost"].sum(): ,.2f}')

        return fited_model
