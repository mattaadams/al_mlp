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


class OnlineActiveLearner(Calculator):
    """Online Active Learner
   Parameters
   ----------
    learner_params: dict
        Dictionary of learner parameters and settings.
        
    trainer: object
        An isntance of a trainer that has a train and predict method.
        
    training_data: list
        A list of ase.Atoms objects that have attached calculators.
        Used as the first set of training data.

    parent_calc: ase Calculator object
        Calculator used for querying training data.
            
    n_ensembles: int.
   	 n_ensemble of models to make predictions.
    
    n_cores: int.
    	 n_cores used to train ensembles.

    parent_calc: ase Calculator object
        Calculator used for querying training data.
        
    base_calc: ase Calculator object
        Calculator used to calculate delta data for training.
        
    trainer_calc: uninitialized ase Calculator object
        The trainer_calc should produce an ase Calculator instance
        capable of force and energy calculations via TrainerCalc(trainer) 
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, learner_params, parent_dataset, parent_calc,base_calc,trainer,trainer_calc, n_ensembles, n_cores
    ):
        Calculator.__init__(self)

        self.n_ensembles = n_ensembles
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.trainer = trainer
        self.trainer_calc_func = trainer_calc
        self.learner_params = learner_params
        self.n_cores = n_cores
        self.ensemble_sets, self.parent_dataset = bootstrap_ensemble(
            parent_dataset, n_ensembles=n_ensembles
        )
        self.ensemble_calc = make_ensemble(
            self.ensemble_sets, self.trainer,self.base_calc, self.n_cores
        )

        self.uncertain_tol = learner_params["uncertain_tol"]
        self.parent_calls = 0
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        energy_pred = self.ensemble_calc.get_potential_energy(atoms)
        force_pred = self.ensemble_calc.get_forces(atoms)
        uncertainty = atoms.info["uncertainty"][0]
        db = connect('dft_calls.db')

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
