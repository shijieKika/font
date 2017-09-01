# -*- coding:utf-8 -*-

class ChineseDict:
    dict_ = dict()
    list_ = []
    def load(self, path):
        self.dict_ = dict()
        self.list_ = []
        with open(path) as fp:
            index = 0
            for line in fp:
                line = line.strip()
                self.dict_[line] = index
                self.list_.append(line)
                index += 1

    def getWord(self, i):
        if i >= 0 and i < len(self.list_):
            return self.list_[i]
        else:
            return None

    def getIndex(self, word):
        if word in self.dict_:
            return self.dict_[word]
        else:
            return None

def main():
    dict_dir = "chinese.dict"
    d = ChineseDict()
    d.load(dict_dir)
    print(d.getWord(6738))
    print(d.getIndex('\xe8\xb2\xac'))

if __name__ == '__main__':
    main()