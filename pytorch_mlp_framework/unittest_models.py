import torch
import unittest
from model_architectures import ConvolutionalProcessingBlockWithBN, ConvolutionalDimensionalityReductionBlockWithBN, ConvolutionalProcessingBlockWithBNRC

class TestConvolutionalProcessingBlockWithBN(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 3, 32, 32)
        self.num_filters = 3
        self.kernel_size = 3
        self.padding = 1
        self.bias = False
        self.dilation = 1

        self.model = ConvolutionalProcessingBlockWithBN(
            input_shape = self.input_shape,
            num_filters = self.num_filters,
            kernel_size = self.kernel_size,
            padding = self.padding,
            bias = self.bias,
            dilation = self.dilation
        )

    def test_forward(self):
        x = torch.randn(self.input_shape)
        output = self.model(x)
        self.assertIsNotNone(output)

        self.assertEqual(output.shape[1], self.num_filters) 
        self.assertEqual(output.shape[2], self.input_shape[2]) 
        self.assertEqual(output.shape[3], self.input_shape[3]) 
        
if __name__ == '__main__':
    unittest.main()

class TestConvolutionalDimensionalityReductionBlockWithBN(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 3, 32, 32)
        self.num_filters = 3
        self.kernel_size = 3
        self.padding = 1
        self.bias = False
        self.dilation = 1
        self.reduction_factor = 2

        self.model = ConvolutionalDimensionalityReductionBlockWithBN(
            input_shape = self.input_shape,
            num_filters = self.num_filters,
            kernel_size = self.kernel_size,
            padding = self.padding,
            bias = self.bias,
            dilation = self.dilation,
            reduction_factor = self.reduction_factor
        )

    def test_forward(self):
        x = torch.randn(self.input_shape)
        output = self.model(x)
        self.assertIsNotNone(output)

        self.assertEqual(output.shape[1], self.num_filters) 
        self.assertEqual(output.shape[2], self.input_shape[2] // self.reduction_factor) 
        self.assertEqual(output.shape[3], self.input_shape[3] // self.reduction_factor)
        
if __name__ == '__main__':
    unittest.main()

class TestConvolutionalProcessingBlockWithBNRC(unittest.TestCase):
    def setUp(self):
        self.input_shape = (1, 3, 32, 32)
        self.num_filters = 3
        self.kernel_size = 3
        self.padding = 1
        self.bias = False
        self.dilation = 1

        self.model = ConvolutionalProcessingBlockWithBNRC(
            input_shape = self.input_shape,
            num_filters = self.num_filters,
            kernel_size = self.kernel_size,
            padding = self.padding,
            bias = self.bias,
            dilation = self.dilation
        )

    def test_forward(self):
        x = torch.randn(self.input_shape)
        output = self.model(x)
        self.assertIsNotNone(output)

        self.assertEqual(output.shape[1], self.num_filters) 
        self.assertEqual(output.shape[2], self.input_shape[2])
        self.assertEqual(output.shape[3], self.input_shape[3]) 

    def test_residual(self):
        x = torch.randn(self.input_shape)
        output = self.model(x)
        
        if self.input_shape[1] != self.num_filters:
            output_res = self.model.layer_dict['identity'].forward(x)
        else:
            output_res = x
        
        residual_output = output - output_res
        self.assertTrue(torch.any(residual_output != 0).item())
        
if __name__ == '__main__':
    unittest.main()