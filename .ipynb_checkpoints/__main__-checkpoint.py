import argparse
import sys
from .bscholes import input_generator, validator, black_scholes_numba


def main():
    pin = input_generator()
    pin = list(pin)
    pin = pin[0]
    in_args = pin['input_args']
    print('The input parameters have been generated, running model...')
    res_one = black_scholes_numba(*in_args)
    out = [i for i in res_one if i[1] <= 100] 
    print(f'The first round of results have been generated.'
          f'{ out }\n'
          f'Beginning Validation...')
    validator(pin['input_args'], pin['input_kwargs'], res_one)
    print('The model successfully ran and was validated!')


if __name__ == "__main__":
    sys.exit(main())
