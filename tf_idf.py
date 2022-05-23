import numpy as np


def tf_idf(bags: np.ndarray) -> np.ndarray:
    # get array of total words per individual text
    totalText = np.sum(bags, axis=1)
    totalText = 1/totalText
    # turn to matrix for multiplication with bag of word matrix
    totalTxt = np.diagflat(totalText)
    tfWords = totalTxt @ bags
    return tfWords


if __name__ == '__main__':
    gg = np.array(
        [
            [1, 0, 3],
            [0, 5, 8]
        ]
    )
    print(tf_idf(gg))
