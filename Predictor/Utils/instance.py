


class Instance(object):
    """
    class for a single line of data, collecting columns
    """
    def __init__(self, columns):
        self.data = {}
        self.collect_fields(columns)

    def collect_fields(self, columns):
        """
        :param fields:  list of columns
        :return:
        """
        for column in columns:
            column_name = column.name
            self.data[column_name] = {}
            for feature in column.data.keys():
                self.data[column_name][feature] = column.data[feature]

    def check(self):
        for column in self.data.keys():
            assert column != {}
            for feature in self.data[column].keys():
                assert feature != []

    def show_data_info(self):
        assert self.data != {}
        print(f'columns : {self.data.keys()}')
        print(f'features: {self.data[self.data.keys()[0]].keys()}')
        print(f'data: {self.data}')