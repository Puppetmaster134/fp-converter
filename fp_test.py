import torch, unittest
from fp import fp8_downcast, uint8_to_fp16

from fp import print_bits
class TestFPTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('Setting up test class.')
        # torch.manual_seed(1)

    def test_expectation(self):
        val = 1 + 1/64
        n_trials = 50
        n_conversions = 256
        t = torch.FloatTensor([val])
        n_bits = 4

        all_sums = []

        for _ in range(n_trials):
            compressed_tensors = []
            for _ in range(n_conversions):
                u1 = fp8_downcast(t,n_bits)
                compressed_tensors.append(uint8_to_fp16(u1,n_bits))
            all_stack = torch.stack(compressed_tensors)
            all_sums.append(torch.sum(all_stack))
        
        std, mean = torch.std_mean(torch.stack(all_sums))

        
        """
        If we run 256 conversions of 1 + 1/64, we should expect 256 * 256/64 as the result (260).
        
        - 1/64 represents the 6th bit of a float32 mantissa or a 1/2**2 chance of adding a bit when converting to fp8
        - torch.rand_like() samples from a uniform distribution [0,1)
        - Repeating the process enough (50) times will see the mean of the results shift towards 260, even in spite of randomness.

        """ 
        self.assertAlmostEqual(mean.item(), val * n_conversions, 0)

    def test_one_overflow(self):
        t = torch.FloatTensor([0.999999940395])
        n_bits = 4

        u1 = fp8_downcast(t,n_bits)
        u2 = uint8_to_fp16(u1,n_bits)

        #The 1-bits should overflow into either of these numbers
        #Used a calculator for this one so go easy on me :)
        self.assertTrue(u2.item() == 1 or u2.item() == 0.96875)

    
    def test_twobit_mantissa(self):
        """Evaluate two bit mantissa. The only fractionals supported in two bits are 0, .25, .50, .75
        """
        t = torch.FloatTensor([1.101])

        mantissa_size = 2
        possible_values = [1.0, 1.25]
        for _ in range(10):
            u1 = fp8_downcast(t,mantissa_size)
            u2 = uint8_to_fp16(u1,mantissa_size)
            self.assertIn(u2.item(), possible_values)

    def test_onebit_mantissa(self):
        """Evaluate two bit mantissa. The only fractionals supported in one bit are 0, .50,
        """
        t = torch.FloatTensor([1.0625])

        mantissa_size = 1
        possible_values = [1,1.5]
        for _ in range(10):
            u1 = fp8_downcast(t,mantissa_size)
            u2 = uint8_to_fp16(u1,mantissa_size)
            self.assertIn(u2.item(), possible_values)
    
    def test_one_bit_exponent(self):
        """Evaluate a one bit exponent (by setting mantissa to 5)
        """
        t = torch.FloatTensor([8.0])

        

        mantissa_size = 5
        possible_values = [1,1.5]
        for _ in range(10):
            u1 = fp8_downcast(t,mantissa_size)
            u2 = uint8_to_fp16(u1,mantissa_size)
            
            # One bit exponent has a bias of -1
            # If the exponent bit is 0, the exponent is 2**-1 or 0.5
            # The 8 bit overflows into the sign position making the number negative
            self.assertEqual(u2.item(), -0.5)

        
if __name__ == '__main__':
    unittest.main()