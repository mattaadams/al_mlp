import os
import copy
import numpy as np

import ase.io
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp

from matplotlib import pyplot as plt



def write_to_db(database, queried_images):
    for image in queried_images:
        database.write(image)


def compute_loss(a, b):
    return np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1)))


def progressive_plot(
    filename, true_relaxed, samples_per_iter, num_iterations, save_to="./"
):
    os.makedirs(save_to, exist_ok=True)
    distance_rmse = []
    data_size = list(range(0, samples_per_iter * num_iterations + 1, samples_per_iter))

    for i in range(len(data_size)):
        ml_relaxed = ase.io.read("{}_iter_{}.traj".format(filename, i), "-1")
        loss = compute_loss(ml_relaxed.positions, true_relaxed.positions)
        distance_rmse.append(loss)
    plt.plot(np.array(data_size) + 1, distance_rmse)
    plt.xlabel("Training Images")
    plt.xticks(np.array(data_size) + 1)
    plt.ylabel("Distance RMSE")
    plt.title("AL Relaxation Learning Curve")
    plt.savefig(save_to + ".png", dpi=300)


def convert_to_singlepoint(images):
    """
    Replaces the attached calculators with singlepoint calculators
    Parameters
    ----------
    images: list
        List of ase atoms images with attached calculators for forces and energies.
    """

    singlepoint_images = []
    cwd = os.getcwd()
    for image in images:
        os.makedirs("./temp", exist_ok=True)
        os.chdir("./temp")
        sample_energy = image.get_potential_energy(apply_constraint=False)
        sample_forces = image.get_forces(apply_constraint=False)
        image.set_calculator(
            sp(atoms=image, energy=sample_energy, forces=sample_forces)
        )
        singlepoint_images.append(image)
        os.chdir(cwd)
        os.system("rm -rf ./temp")

    return singlepoint_images

def compute_with_calc(images, calculator):
    """
    Calculates forces and energies of images with calculator.
    Returned images have singlepoint calculators.
    Parameters
    ----------
    images: list
        List of ase atoms images to be calculated.
    calc: ase Calculator object
        Calculator used to get forces and energies.
    """

    singlepoint_images = []
    cwd = os.getcwd()
    for image in images:
        os.makedirs("./temp", exist_ok=True)
        os.chdir("./temp")
        image.set_calculator(calculator)
        print(image)
        sample_energy = image.get_potential_energy(apply_constraint=False)
        sample_forces = image.get_forces(apply_constraint=False)
        image.set_calculator(
            sp(atoms=image, energy=sample_energy, forces=sample_forces)
        )
        singlepoint_images.append(image)
        os.chdir(cwd)
        os.system("rm -rf ./temp")

    return singlepoint_images
