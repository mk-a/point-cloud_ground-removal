import laspy
import numpy as np
from pathlib import Path
import argparse
from numba import njit


@njit
def get_mask(pc, min_x, min_y, d_x, d_y, i, j):
    return np.argwhere(
        (pc[:, 0] >= min_x + i * d_x)
        & (pc[:, 0] < min_x + (i + 1) * d_x)
        & (pc[:, 1] >= min_y + j * d_y)
        & (pc[:, 1] < min_y + (j + 1) * d_y)
    ).flatten()


def main():
    in_path = Path(args.in_file)
    if args.out_file is None:
        out_path = in_path.with_name(in_path.stem + "_no_ground.las")
    else:
        out_path = args.out_file
    las = laspy.file.File(in_path, mode="r")
    pc = np.column_stack((las.x, las.y, las.z))
    min_x, min_y, min_z = pc.min(axis=0)
    max_x, max_y, max_z = pc.max(axis=0) + 1e-3  # add constant to include border points
    # d_x and d_y defines the size of the subdivisions
    # d_z is the difference betzeen the global max and min of height
    d_x, d_y, d_z = pc.max(axis=0) - pc.min(axis=0)
    d_x /= args.x
    d_y /= args.y
    ground = np.full(len(las), args.c_n_steps, dtype=np.uint16)
    pt_src_id = np.zeros(len(las), dtype=np.uint16)
    n = 0
    c_values = np.linspace(args.c_max, args.c_min, args.c_n_steps)
    print(list(zip(range(args.c_n_steps - 1, -1, -1), c_values)))
    for i in range(args.x):
        for j in range(args.y):
            mask = get_mask(pc, min_x, min_y, d_x, d_y, i, j)
            if mask.size == 0:
                continue
            print("{:03d} {:03d}".format(i, j), end="\r")
            local_pc = pc[mask]
            local_min = local_pc.min(axis=0)[2]
            local_max = local_pc.max(axis=0)[2]
            C_list = []
            for c in c_values:
                C_list.append(local_min + d_z * (1 - (max_z - local_max) / d_z) * c)
            for k, C_he in zip(range(9, -1, -1), C_list):
                ground[mask[local_pc[:, 2] <= C_he]] = k
            pt_src_id[mask] = n
            n += 1
    las2 = laspy.file.File(out_path, mode="w", header=las.header)
    las2.points = las.points
    las2.Classification = ground
    las2.pt_src_id = pt_src_id
    las.close()
    las2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute Jaromír Landa's ground removal algorithm
    http://dx.doi.org/10.11118/actaun201361072415"""
    )
    parser.add_argument("in_file", type=str, help="input file to compute")
    parser.add_argument(
        "-o", "--out_file", type=str, default=None, help="output file. Optional"
    )
    parser.add_argument(
        "-x", type=int, default=50, help="number of division along the x axis"
    )
    parser.add_argument(
        "-y", type=int, default=50, help="number of division along the y axis"
    )
    parser.add_argument(
        "-c_max", type=float, default=0.030, help="height coefficient max value"
    )
    parser.add_argument(
        "-c_min", type=float, default=0.001, help="height coefficient min value"
    )
    parser.add_argument(
        "-c_n_steps",
        type=int,
        default=10,
        help="height coefficient number of steps value",
    )
    parser.add_argument(
        "-c", type=float, default=0.015, help="height coefficient value"
    )
    args = parser.parse_args()
    main()
