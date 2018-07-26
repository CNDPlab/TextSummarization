import pyhanlp


def segment(data_str):
    res = list(pyhanlp.HanLP.segment(data_str))
    seg = [i.toString().split('/')[0] for i in res]
    pos = [i.toString().split('/')[1] for i in res]
    return seg, pos

class Column_base(object):

    def __init__(self, name, padding_sign=True):
        self.name = name
        self.data = {}
        self.padding_sign = 'padding_sign'
        self.data[self.padding_sign] = padding_sign

    def take_in_data(self, data):

        raise NotImplementedError

    def process_func(self, data):

        raise NotImplementedError

    def show_data_info(self):
        assert self.data != {}
        print(f'features : {self.data.keys()}')
        print(f'data: {self.data}')



class TextColumn(Column_base):
    def __init__(self, name, padding_sign=True):
        super(TextColumn, self).__init__(name, padding_sign)

    def take_in_data(self, data):
        self.data['seg'], self.data['pos'] = self.process_func(data)

    def process_func(self, data):
        seg, pos = segment(data)
        return seg, pos

def test_textcolumn():
    textcolumn = TextColumn('text',True)
    data = '我爱北京天安门，天安门山太阳升'
    textcolumn.take_in_data(data)
    textcolumn.show_data_info()

def test():
    test_textcolumn()


if __name__ == '__main__':
    test()