# -*- coding:utf-8 -*-

class ChineseDict:
    dict_ = dict()
    list_ = []

    def __init__(self, path):
        self.dict_ = dict()
        self.list_ = []
        with open(path) as fp:
            index = 0
            for line in fp:
                line = line.strip()
                self.dict_[line] = index
                self.list_.append(line)
                index += 1

    def size(self):
        return len(self.list_)

    def get_word(self, i):
        if i >= 0 and i < len(self.list_):
            return self.list_[i]
        else:
            return None

    def get_index(self, word):
        if word in self.dict_:
            return self.dict_[word]
        else:
            return None


def main():
    dict_dir = "/Users/msj/Code/font/font/static/chinese.dict"
    d = ChineseDict(dict_dir)
    print(d.get_word(6738))
    print(d.get_index('\xe8\xb2\xac'))


if __name__ == '__main__':
    main()
