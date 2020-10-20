import os
import sys
import copy
import numpy as np
import pandas as pd
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.calculators.calculator import Calculator

from al_mlp.bootstrap import bootstrap_ensemble
from al_mlp.trainer import ensemble_trainer


__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class AMPOnlineCalc(Calculator):
    """Atomistics Machine-Learning Potential (AMP) ASE calculator
   Parameters
   ----------
    model : object
        Class representing the regression model. Input arguments include training
        images, descriptor type, and force_coefficient. Model structure and training schemes can be
        modified directly within the class.

    label : str
        Location to save the trained model.

    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, parent_dataset, parent_calc, n_ensembles, n_cores, training_params
    ):
        Calculator.__init__(self)

        self.n_ensembles = n_ensembles
        self.parent_calc = parent_calc
        self.training_params = training_params
        self.n_cores = n_cores
        self.ensemble_sets, self.parent_dataset = bootstrap_ensemble(
            parent_dataset, n_ensembles=n_ensembles
        )
        self.ensemble_calc = ensemble_trainer(
            self.ensemble_sets, self.training_params, self.n_cores
        )

        self.uncertain_tol = training_params["uncertain_tol"]
        self.parent_calls = 0
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        energy_pred = self.ensemble_calc.get_potential_energy(atoms)
        force_pred = self.ensemble_calc.get_forces(atoms)
        uncertainty = atoms.info["uncertainty"][0]
        db = connect('dft_calls.db')
        new_data = atoms.copy()
        new_data.set_calculator(copy.copy(self.parent_calc))
        cwd = os.getcwd()
        if uncertainty >= self.uncertain_tol:
            print('DFT required')
            new_data = atoms.copy()
            new_data.set_calculator(copy.copy(self.parent_calc))
           # os.makedirs("./temp", exist_ok=True)
           # os.chdir("./temp")

            energy_pred = new_data.get_potential_energy(apply_constraint=False)
            force_pred = new_data.get_forces(apply_constraint=False)
            new_data.set_calculator(
                sp(atoms=new_data, energy=energy_pred, forces=force_pred)
            )
           # os.chdir(cwd)
           # os.system("rm -rf ./temp")

            energy_list.append(energy_pred)
            db.write(new_data)
            self.ensemble_sets, self.parent_dataset = bootstrap_ensemble(
                self.parent_dataset, self.ensemble_sets, new_data=new_data
            )

            self.ensemble_calc = ensemble_trainer(
                self.ensemble_sets, self.training_params, self.n_cores
            )
            self.parent_calls += 1
        else:
            db.write(None)
        self.results["energy"] = energy_pred
        self.results["forces"] = force_pred
