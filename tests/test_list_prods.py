from pit import list_prods

# @pytest.mark.parametrize(
#     ''
# )
def test_prods_complete():
    valid_prods = [
        '0930',
        '0930_1h',
        '1000',
        '1000_1h',
        '1030',
        '1030_1h',
        '1100',
        '1100_1h',
        '1300',
        '1300_1h',
        '1330',
        '1330_1h',
        '1400',
        '1400_1h',
        '1430',
        '1430_30m',
    ]
    assert set(list_prods()) == set(valid_prods)