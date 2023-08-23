import sonecules as sn


def test_startup():
    sn.startup()
    assert isinstance(sn.gcc(), sn.Context)
