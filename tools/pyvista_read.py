import sys
import pyvista as pv
import numpy as np


def main():
    f = sys.argv[1]
    if f.endswith('.npy'):
        a = np.load(f)[:, :3]
        pd = pv.PolyData(a)
        pd.plot()
    else:
        pl = pv.Plotter()
        try:
            pl.add_mesh(pv.read(f), rgb=True, point_size=5)
            pl.show()
        except Exception as e:
            pl.add_mesh(pv.read(f), point_size=5)
            pl.show()

if __name__ == '__main__':
    main()
