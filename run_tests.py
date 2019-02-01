import unittest
loader = unittest.TestLoader()
tests = loader.discover('tests')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)