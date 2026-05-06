from parsimmon.study import Study


def test_arange():
    assert list(Study.arange(0, 6, 2)) == [0, 2, 4]
    assert list(Study.arange(3)) == [0, 1, 2]


def test_each():
    r = Study.each([1, 4, 8])
    assert list(r) == [1, 4, 8]
    assert len(r) == 3
