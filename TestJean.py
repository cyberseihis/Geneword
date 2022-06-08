from Jean import main


def test():
    jj = main()
    jg = next(jj)
    jg = next(jj)
    rep = jg()


if __name__ == '__main__':
    test()
