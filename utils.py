def print_tensor_info(tensor, name="tensor"):
    print(f'{name:>25}\t{tensor.dtype}\t{tensor.shape}\t[{tensor.min()}, {tensor.max()}]')
