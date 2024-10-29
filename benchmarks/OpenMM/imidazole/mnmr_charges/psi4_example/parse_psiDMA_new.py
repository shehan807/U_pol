#!/usr/bin/env python
import sys

#******************** python script for psi4 dma output to pull multipole moments
#******************* 
#*******************   IMPORTANT:  Units should all be A.U., we need to convert coordinates from Angstrom to Bohr
conv=1.88973;

ifile = sys.argv[1]
if len(sys.argv) > 2:
    xyzfile = sys.argv[2]

data=[]
match=False
with open(ifile) as f:
    while True:
        line = f.readline()
        if "Multipole moments" in line:
            match=True
        if match:
            data.append( line.rstrip() ) 
        if not line:
            break

multipole_moments=[]
atom_xyz=[]
atom_multipole={}
for line in data:
    # break if total multipoles section
    if "Total multipoles" in line:
        break
    # if new element block
    if "x =" in line:
        # push last multipole dictionary to list, and reinitialize
        multipole_moments.append( atom_multipole )
        atom_multipole={}
        atomdata = line.split()
        atom_xyz.append( [ atomdata[0] , float(atomdata[3]) , float(atomdata[6]) , float(atomdata[9]) ] )
    elif "Maximum rank" in line:
        pass
    else:
         # this is a multipole line.  Multipole entires come in 3's as split by whitespace,
         # e.g. Q11c = 0.14683 .  On splitting on whitespace, should get a multiple of three entries
         multipoles = line.rstrip().split()
         for i in range( len(multipoles ) ) :
             #print( multipoles )
             # loop over every third, add to dictionary
             if i % 3 == 0 :
                 #print( multipoles[i] , multipoles[i+2] )
                 # add to dictionary
                 atom_multipole[ multipoles[i] ] = multipoles[i+2]

# append moments for last atom
multipole_moments.append( atom_multipole )
# pop off first junk element
multipole_moments.pop(0)

#for multipoles in multipole_moments:
#    print( 'Multipoles for atom' )
#    for key,value in multipoles.items():
#        print("Key : {} , Value : {}".format(key,value))

# returns multipole moment if it exists, or 0.0
def get_moment( multipoles , key ):
    if key in multipoles:
        return float(multipoles[key])
    else:
        return 0.0

# now print formatted multipole file
for i in range(len(atom_xyz)):
    # unit conversion angstrom to bohr ...
    print("{0:3s}{1:16.6f}{2:16.6f}{3:16.6f}{4:7d}".format( atom_xyz[i][0] ,atom_xyz[i][1]*conv ,atom_xyz[i][2]*conv ,atom_xyz[i][3]*conv , 4 ))
    # print block of moments ...
    # rank 0 block
    print("{0:9.6f}".format( get_moment( multipole_moments[i] , 'Q00' ) ) )
    # rank 1 block
    print("{0:9.6f}\t{1:9.6f}\t{2:9.6f}".format( get_moment( multipole_moments[i] , 'Q10' ) , get_moment( multipole_moments[i] , 'Q11c' ) , get_moment( multipole_moments[i] , 'Q11s' )  ) )
   # rank 2 block
    print("{0:9.6f}\t{1:9.6f}\t{2:9.6f}\t{3:9.6f}\t{4:9.6f}".format( get_moment( multipole_moments[i] , 'Q20' ) , get_moment( multipole_moments[i] , 'Q21c' ) , get_moment( multipole_moments[i] , 'Q21s' ) , get_moment( multipole_moments[i] , 'Q22c' ) , get_moment( multipole_moments[i] , 'Q22s' ) ) )
   # rank 3 block
    print("{0:9.6f}\t{1:9.6f}\t{2:9.6f}\t{3:9.6f}\t{4:9.6f}\t{5:9.6f}\t{6:9.6f}".format( get_moment( multipole_moments[i] , 'Q30' ) , get_moment( multipole_moments[i] , 'Q31c' ) , get_moment( multipole_moments[i] , 'Q31s' ) , get_moment( multipole_moments[i] , 'Q32c' ) , get_moment( multipole_moments[i] , 'Q32s' ) , get_moment( multipole_moments[i] , 'Q33c' ) , get_moment( multipole_moments[i] , 'Q33s' ) ) )
  # rank 4 block
    print("{0:9.6f}\t{1:9.6f}\t{2:9.6f}\t{3:9.6f}\t{4:9.6f}\t{5:9.6f}\t{6:9.6f}\t{7:9.6f}\t{8:9.6f}".format( get_moment( multipole_moments[i] , 'Q40' ) , get_moment( multipole_moments[i] , 'Q41c' ) , get_moment( multipole_moments[i] , 'Q41s' ) , get_moment( multipole_moments[i] , 'Q42c' ) , get_moment( multipole_moments[i] , 'Q42s' ) , get_moment( multipole_moments[i] , 'Q43c' ) , get_moment( multipole_moments[i] , 'Q43s' ) , get_moment( multipole_moments[i] , 'Q44c' ) , get_moment( multipole_moments[i] , 'Q44s' ) ) )


# if xyz file was input, print an xyz coordinate file
if len(sys.argv) > 2 :
    with open( xyzfile , 'w' ) as f:
        f.write( str(len(atom_xyz)) + "\n" )
        f.write( 'optimized coordinates in Angstrom\n' )
        for atom in atom_xyz:
              f.write("{0:3s}{1:16.6f}{2:16.6f}{3:16.6f}\n".format( atom[0] ,atom[1] , atom[2] , atom[3] ))



