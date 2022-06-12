from Jean import main
import pickle


def test():
    ch, me = main()
    print(ch)
    print(len(me))


if __name__ == '__main__':
    ch, me = main()
    with open("genetics.pkl", 'wb') as fi:
        pickle.dump((ch, me), fi)
