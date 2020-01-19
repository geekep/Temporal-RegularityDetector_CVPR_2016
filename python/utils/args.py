import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dst-path', type=str, required=True, help='destination to store videos')
parser.add_argument('--urls-path', type=str, required=True, help='path to urls file')
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()

