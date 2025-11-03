import os
import sys
import unittest



if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))

    test_modules = ["test_rotations", "test_query"]

    all_tests = []
    for module in test_modules:
        all_tests.extend(__import__(module).tests)

    for test in all_tests:
        print("*" * 20)
        unittest.TextTestRunner(verbosity=2).run(
            unittest.TestLoader().loadTestsFromTestCase(test)
        )
        print("*" * 20)
