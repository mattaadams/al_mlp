from al_mlp.calcs import DeltaCalc
from ase.calculators.emt import EMT
from ase.calculators.morse import MorsePotential
import numpy as np
import ase
import copy
from al_mlp.learner import OfflineActiveLearner
from al_mlp.calcs import TrainerCalc
from ase.calculators.emt import EMT
from ase.calculators.morse import MorsePotential
from ase import Atoms
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS, QuasiNewton
from ase.build import bulk
from ase.utils.eos import EquationOfState
parent_calculator = EMT()
energies = []
volumes = []
LC = [3.5, 3.55, 3.6, 3.65, 3.7, 3.75]

for a in LC:
   cu_bulk = bulk('Cu', 'fcc', a=a)
   calc = EMT()
   cu_bulk.set_calculator(calc)
   e = cu_bulk.get_potential_energy()
   energies.append(e)
   volumes.append(cu_bulk.get_volume())


eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
aref=3.6
vref = bulk('Cu', 'fcc', a=aref).get_volume()
copper_lattice_constant = (v0/vref)**(1/3)*aref
slab = fcc100("Cu", a=copper_lattice_constant, size=(2, 2, 3))
ads = molecule("C")
add_adsorbate(slab, ads, 2, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in slab if (atom.tag == 3)])
slab.set_constraint(cons)
slab.center(vacuum=13.0, axis=2)
slab.set_pbc(True)
slab.wrap(pbc=[True] * 3)
slab.set_calculator(copy.copy(parent_calculator))
slab.set_initial_magnetic_moments()
image_copy = slab
# create image with base calculator attached
base_calc = MorsePotential()
image_copy.set_calculator(base_calc) 

#add
delta_calc = DeltaCalc([parent_calculator,base_calc],"add",[slab,image_copy])
#Set slab calculator to delta calc and evaluate energy
slab.set_calculator(delta_calc)
slab.get_potential_energy()

#Sub
delta_calc = DeltaCalc([parent_calculator,base_calc],"sub",[slab,image_copy])
#Set slab calculator to delta calc and evaluate energy
slab.set_calculator(delta_calc)
slab.get_potential_energy()
