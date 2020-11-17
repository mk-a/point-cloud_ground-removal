import laspy
import numpy as np
from pathlib import Path
import argparse


def main():
    in_path = Path(args.in_file)
    if args.out_file is None:
        out_path = in_path.with_name(in_path.stem + "_no_ground.las")
    else:
        out_path = args.out_file
    las = laspy.file.File(in_path, mode="r")
    pc = np.column_stack((las.x, las.y, las.z))
    min_x, min_y, min_z = pc.min(axis=0)
    max_x, max_y, max_z = pc.max(axis=0) + 1e-3
    d_x, d_y, d_z = pc.max(axis=0) - pc.min(axis=0)
    d_x /= args.x
    d_y /= args.y
    ground = np.full(len(las), False)
    pt_src_id = np.zeros(len(las), dtype=np.uint16)
    ce = 0.015
    n = 0
    for i in range(args.x):
        for j in range(args.y):
            mask = np.argwhere(
                (pc[:, 0] >= min_x + i * d_x)
                & (pc[:, 0] < min_x + (i + 1) * d_x)
                & (pc[:, 1] >= min_y + j * d_y)
                & (pc[:, 1] < min_y + (j + 1) * d_y)
            ).flatten()
            if mask.size == 0:
                continue
            local_pc = pc[mask]
            local_min = local_pc.min(axis=0)[2]
            local_max = local_pc.max(axis=0)[2]
            C_he = local_min + d_z * (1 - (max_z - local_max) / d_z) * ce
            print(
                "{:3d} {:3d}\t{:.1f} {:.1f} {:.1f}".format(
                    i, j, local_min, C_he, local_max
                ),
                end="\r",
            )
            # print(i, j)
            # print(
            #     "l_min: {:.1f}\tl_max: {:.1f}\tmax_z: {:.1f}".format(
            #         local_min, local_max, max_z
            #     )
            # )
            # print("(max_z - local_max)", (max_z - local_max))
            # print("(max_z - local_max) / d_z", (max_z - local_max) / d_z)
            # print("d_z * (1 - (max_z - local_max) / d_z)", d_z * (1 - (max_z - local_max) / d_z))
            # print("C_he", C_he)

            ground[mask[local_pc[:, 2] <= C_he]] = True
            pt_src_id[mask[local_pc[:, 2] <= C_he]] = n
            n += 1
    las2 = laspy.file.File(out_path, mode="w", header=las.header)
    las2.points = las.points
    las2.Classification[ground] = 2
    las2.Classification[np.logical_not(ground)] = 1
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
    args = parser.parse_args()
    main()
