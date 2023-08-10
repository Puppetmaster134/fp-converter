# fp-converter
Convert Pytorch FP32, FP16, and BFloat16 to FP8 and back again

There are two main functions here:

`fp8_downcast(source_tensor : torch.Tensor, n_bits : int)`

`fp8_downcast` expects a source Pytorch tensor of either Float32, Float16, or BFloat16. This tensor is downcasted to Float8 using bit shifting and stochastic rounding. 
The resulting tensor is returned with the dtype `torch.uint8` so it can be loaded onto devices that don't support 8 bit floating point values.
Notably, this function takes an argument `n_bits` which allocates n bits of the target tensor for the mantissa of the floating point value.


`uint8_to_fp16(source_tensor : torch.ByteTensor, n_bits : int)`

`uint8_to_fp16` expects a `torch.uint8` tensor produced by the first function. Using bitshifting this tensor will be converted to Float16.
This function also expects an argument `n_bits` denoting the mantissa length of the source tensor.
