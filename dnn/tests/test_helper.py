from ..helper import *
import unittest


class HelperTest(unittest.TestCase):
    def testSigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)




if __name__ == '__main__':
    unittest.main()
