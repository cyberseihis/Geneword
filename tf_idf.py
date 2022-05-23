#!/usr/bin/env python3
import numpy as np


def tf_idf(bags: np.ndarray) -> np.ndarray:
    # get array of total words per individual text
    totalText = np.sum(bags, axis=1)
    totalText = 1/totalText
    # turn to matrix for multiplication with bag of word matrix
    totalTxt = np.diagflat(totalText)
    tf = totalTxt @ bags
    # number of texts each word appears in
    appears = np.count_nonzero(bags, axis=0)
    N = np.size(bags, axis=0)
    idf = np.log2(N) - np.log2(appears)
    tfidf = tf @ np.diagflat(idf)
    return tfidf


if __name__ == '__main__':
    gg = np.array(
        [
            [1, 0, 3],
            [0, 5, 8]
        ]
    )
    print(tf_idf(gg))
