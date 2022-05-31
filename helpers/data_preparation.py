import pandas as pd

class DataTransformation():
    def __init__(self, dataset: pd.DataFrame()):
        self.dataset = dataset

    def filter_transaction_type(self):
        print('filtrando o dataset...',
        'selecionando somente as transações do tipo saque e transferencia')
        transaction_type = ['TRANSFER', 'CASH_OUT']

        self.dataset = self.dataset.query('type in @transaction_type')

        print('dataset filtrado')
        return self.dataset
    
    def drop_useless_columns(self):
        useless_columns = ['nameOrig', 'nameDest', 'isFlaggedFraud']
        print('eliminando as colunas...')
        self.dataset.drop(useless_columns, axis=1, inplace=True)

        print('ok')
        return self.dataset

    def convert_step_to_hour(self):
        print('convertendo step para hora...')
        self.dataset['hour'] = self.dataset['step'] % 24
        self.dataset.drop('step', axis=1, inplace=True)
        print('ok')

    def reorder_columns(self):
        print('ordenando as colunas...')
        columns_order = list(self.dataset.columns)
        
        # using insert() + pop()
        # # shift last element to first
        columns_order.insert(0, columns_order.pop())
        self.dataset = self.dataset.reindex(columns=columns_order)

        print('ok')

    def transform_data(self):
        self.dataset = self.filter_transaction_type()
        self.dataset = self.drop_useless_columns()
        self.convert_step_to_hour()
        self.reorder_columns()


        print('dataset pronto pra utilização')
        print('tamanho do dataset: ', self.dataset.shape)
        return self.dataset