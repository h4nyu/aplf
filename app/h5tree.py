import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from argparse import ArgumentParser
import h5py

def main():
    args = parse_args()
    with h5py.File(args.file, "r") as h5file:
        print_h5(h5file, 0)

def print_h5(node, depth):
    indent = "    "
    node_name = node.name.split("/")[-1]
    if node_name == "":
        node_name = "/"
    if isinstance(node, h5py.Dataset):
        print(f"{indent * depth}{node_name} (dtype={node.dtype}, shape={node.shape})")
    else:
        print(f"{indent * depth}{node_name}")
    if isinstance(node, h5py.File) or isinstance(node, h5py.Group):
        for key in node.keys():
            print_h5(node[key], depth + 1)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file", help="hdf5 file (*.h5)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
