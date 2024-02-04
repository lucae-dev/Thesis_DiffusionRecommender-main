try:
    import Cython
    print("Cython is installed, version:", Cython.__version__)
except ImportError:
    print("Cython is not installed in this environment.")
