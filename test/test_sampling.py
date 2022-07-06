import unittest
import numpy as np
from autolabel.dataset import IndexSampler


class SamplingTestCase(unittest.TestCase):

    def test_find_classes(self):
        semantics = np.zeros((2, 10), int)
        sampler = IndexSampler()
        sampler.update(semantics)
        self.assertFalse(sampler.has_semantics)

        self.assertEqual(len(sampler.classes), 0)
        semantics[0, 5] = 1
        semantics[0, 6] = 2
        sampler.update(semantics)
        self.assertEqual(len(sampler.classes), 2)
        self.assertEqual(sampler.classes[0], 1)
        self.assertEqual(sampler.classes[1], 2)

    def test_sampling(self):
        semantics = np.zeros((2, 10), int)
        semantics[0, 5] = 1
        semantics[0, 0] = 2
        semantics[1, 5] = 3
        sampler = IndexSampler()
        sampler.update(semantics)
        random_class = sampler.sample_class()
        self.assertIn(random_class, [1, 2, 3])

        random_image, random_index = sampler.sample(1, 1)
        self.assertEqual(random_image, 0)
        self.assertEqual(random_index[0], 5)
        random_image, random_index = sampler.sample(2, 1)
        self.assertEqual(random_image, 0)
        self.assertEqual(random_index[0], 0)

        random_image, random_indices = sampler.sample(3, 5)
        self.assertEqual(random_image, 1)
        self.assertEqual(len(random_indices), 5)
        self.assertEqual(np.random.choice(random_indices), 5)
        self.assertTrue(sampler.has_semantics)

    def test_semantic_indices(self):
        semantics = np.zeros((5, 10), int)
        semantics[0, 5] = 1
        semantics[2, 0] = 2
        semantics[4, 5] = 3
        sampler = IndexSampler()
        sampler.update(semantics)
        indices = sampler.semantic_indices()
        self.assertEqual(len(indices), 3)
        self.assertIn(0, indices)
        self.assertIn(2, indices)
        self.assertIn(4, indices)


if __name__ == "__main__":
    unittest.main()
