import argparse

parser = argparse.ArgumentParser()

# String argument
parser.add_argument('--name',   type=str,   help='Your name')

# Integer argument
parser.add_argument('--age',    type=int,   help='Your age')

# Float argument
parser.add_argument('--score',  type=float, help='Your score')

# Boolean flag (True/False)
parser.add_argument('--verbose', action='store_true', help='Print details')

args = parser.parse_args()

print(args.name)
print(args.age)
print(args.score)
print(args.verbose)