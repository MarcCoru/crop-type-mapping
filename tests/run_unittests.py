import unittest
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--discover-pattern',default="test*.py", help='unittest discover pattern. '
                                                                       'can be used to seletively run some unittests...')
args = parser.parse_args()

loader = unittest.TestLoader()
suite = loader.discover("tests", pattern=args.discover_pattern)

runner = unittest.TextTestRunner()
runner.run(suite)