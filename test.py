# -*- coding:utf-8 -*-

from utils.chinese_dict import ChineseDict

def far():
    dict_dir = "./chinese.dict"
    d = ChineseDict()
    d.load(dict_dir)
    print(d.getWord(8876))
    print(d.getIndex('一'))

def main():
    iter([])




if __name__ == '__main__':
    main()

