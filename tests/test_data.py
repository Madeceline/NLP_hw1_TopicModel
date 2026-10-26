import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.data.preprocess import bounded_counts


def test_train_only_vocabulary_and_token_cap():
    vectorizer = CountVectorizer().fit(['apple apple pear', 'pear fruit'])
    counts, _ = bounded_counts(vectorizer, ['apple unseen unseen pear '*20], 9, 123)
    assert 'unseen' not in vectorizer.vocabulary_
    assert counts.sum() == 9
    again, _ = bounded_counts(vectorizer, ['apple unseen unseen pear '*20], 9, 123)
    np.testing.assert_array_equal(counts.toarray(), again.toarray())
