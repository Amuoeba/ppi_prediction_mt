#Imports from external libraries
import unittest
import os
#Imports from internal libraries
import utils



class TestCompareFiles(unittest.TestCase):

    def test_same(self):
        a = "/home/erikj/projects/insidrug/py_proj/erikj/test_samples/mock_files/fileA.txt"
        b = "/home/erikj/projects/insidrug/py_proj/erikj/test_samples/mock_files/fileB.txt"
        c = "/home/erikj/projects/insidrug/py_proj/erikj/test_samples/mock_files/fileC.txt"
        with self.subTest():
            self.assertEqual(utils.compare_files(a,a),True,"Should be True")
        with self.subTest():
            self.assertEqual(utils.compare_files(a,b),True,"Should be True")
        with self.subTest():
            self.assertEqual(utils.compare_files(a,c),False,"Should be False")
        
        

if __name__ == '__main__':
    unittest.main()

