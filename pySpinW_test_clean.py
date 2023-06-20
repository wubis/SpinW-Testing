import libpymcr
import numpy as np

np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(
        float=lambda x: f"{x:8.4f}",
        complexfloat = lambda x: f"{x.real:8.4f} {'-' if x.imag<0 else '+'} {abs(x.imag):8.4f}",
    )
)

m = libpymcr.Matlab('C:/Users/jyw1/Desktop/SpinW/spinw/PySpinW.ctf')

# Create a spinw model, in this case a triangular antiferromagnet
s = m.sw_model('triAF', 1)

# Specify the start and end points of the q grid and the number of points

q_start = [0, 0, 0]

q_end = [1, 1, 0]

pts = 501  # Can change this value to a smaller number to better visualize the data

# Calculate the spin wave spectrum

spec = m.spinwave(s, [q_start, q_end, pts])

FMspec = m.sw_neutron(spec)
FMspec = m.sw_egrid(FMspec,'component','Sperp')

# Uncomment for figure; however, second half of the code does not run if uncommented

"""m.figure()
m.subplot(2,1,1)
m.sw_plotspec(FMspec,'mode',1,'colorbar',False)
m.axis([0, 1, 0, 5])
m.subplot(2,1,2)
m.sw_plotspec(FMspec,'mode',2)
m.axis([0, 1, 0, 2])
m.swplot.subfigure(1,3,1)"""

def convert():
    s = 'Sab'
    from scipy.io import loadmat
    mat = loadmat("C:/Users/jyw1/Desktop/SpinW/spinw/Sab.mat")
    print(mat[s])
    return mat[s]

matlab_Sab = convert()
python_Sab = spec["Sab"]

delta = matlab_Sab - python_Sab
peak = np.argmax(abs(delta))

absolute = abs(delta)

print("absolute delta", np.max(absolute))
