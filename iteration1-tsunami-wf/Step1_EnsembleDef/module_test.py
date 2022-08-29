# test_mock_argparse.py
import argparse
try:
    from unittest import mock  # python 3.3+
except ImportError:
    import mock  # python 2.6-3.2


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args)  # NOTE: this is how you would check what the kwargs are if you're unsure
    return args.accumulate(args.integers)


@mock.patch('argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(accumulate=sum, integers=[1,2,3]))
def test_command(mock_args):
    res = main()
    assert res == 6, "1 + 2 + 3 = 6"

