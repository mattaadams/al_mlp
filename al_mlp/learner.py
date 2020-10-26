import random
import os
#from amptorch.ase_utils import AMPtorch
#from amptorch.trainer import AtomsTrainer
from al_mlp.calcs import DeltaCalc
from al_mlp.al_utils import convert_to_singlepoint, compute_with_calc

import ase.db


class OfflineActiveLearner:
    """Offline Active Learner

    Parameters
    ----------

    learner_settings: dict
        Dictionary of learner parameters and settings.
        
    trainer: object
        An isntance of a trainer that has a train and predict method.
        
    training_data: list
        A list of ase.Atoms objects that have attached calculators.
        Used as the first set of training data.

    parent_calc: ase Calculator object
        Calculator used for querying training data.
        
    base_calc: ase Calculator object
        Calculator used to calculate delta data for training.
        
    ensemble: boolean.
    Whether to train an ensemble of models to make predictions. ensemble
    must be True if uncertainty based query methods are to be used. 
     """
    
    def __init__(self, learner_params, trainer, training_data, parent_calc, base_calc,ensemble=False):
        self.learner_params = learner_params
        self.trainer = trainer
        self.training_data = training_data
        self.parent_calc = parent_calc
        self.base_calc = base_calc
        self.calcs = [parent_calc, base_calc]
        self.ensemble = ensemble
        self.init_training_data()
        self.iteration = 0
        if ensemble:
             assert isinstance(ensemble,ent) and ensemble > 1, "Invalid ensemble!"
             self.training_data, self.parent_dataset = bootstrap_ensemble(
                self.training_data, n_ensembles=ensemble
             )
        else:
             self.parent_dataset = self.training_data 
    def init_training_data(self):
        """
        Prepare the training data by attaching delta values for training.
        """
        #print(self.training_data) 
        raw_data = self.training_data
        #print(raw_data)
        sp_raw_data = convert_to_singlepoint(raw_data)
        parent_ref_image = sp_raw_data[0].copy()
        base_ref_image = compute_with_calc(sp_raw_data[:1],self.base_calc)[0]
        self.refs = [parent_ref_image, base_ref_image]
        self.delta_sub_calc = DeltaCalc(self.calcs, "sub", self.refs)
        self.training_data = compute_with_calc(sp_raw_data, self.delta_sub_calc)
        
    def learn(self, atomistic_method):
        """
        Conduct offline active learning. Returns the trained calculator.
        
        Parameters
        ----------

        atomistic_method: object
            Define relaxation parameters and starting image.
        """
        max_iterations = self.learner_params["max_iterations"]
        samples_to_retrain = self.learner_params["samples_to_retrain"]
        filename = self.learner_params["filename"]
        file_dir = self.learner_params["file_dir"]
        queries_db = ase.db.connect("{}.db".format(filename))
        os.makedirs(file_dir, exist_ok=True)
        self.iteration = 0
        terminate = False
        while not terminate:
            fn_label = f"{file_dir}{filename}_iter_{self.iteration}"
            if self.iteration > 0:
                self.query_data(sample_candidates)
                
            self.trainer.train(self.training_data)
            trainer_calc = self.make_trainer_calc()
            trained_calc = DeltaCalc([trainer_calc, self.base_calc], "add", self.refs)
            # run atomistic_method using trained ml calculator  
            atomistic_method.run(calc=trained_calc, filename=fn_label)
            #collect trajectory file
            sample_candidates = atomistic_method.get_trajectory(filename=fn_label)
            
            terminate = self.check_terminate()
            self.iteration += 1
            
        return trained_calc
            
    def query_data(self, sample_candidates,samples_to_retrain):
        """
        Queries data from a list of images. Calculates the properties and adds them to the training data.
        
        Parameters
        ----------

        sample_candidates: list
            List of ase atoms objects to query from.
        """
        queried_images = self.query_func(sample_candidates,sample_to_retrain)
        for image in queried_images:
            image.calc = None
        queried_images = compute_with_calc(queried_images,self.delta_sub_calc)
        self.training_data += queried_images
        
    
    def check_terminate(max_iterations):
        """
        Default termination function. Teminates after a specified number of iterations.
        """
        if self.iterations >= max_iterations:
            return True
        return False
        
    def query_func(sample_candidates,samples_to_retrain):
        """
        Detault query strategy. 
        """
        queried_images = random.sample(sample_candidates,samples_to_retrain)
        return queried_images
       
    def make_trainer_calc(self):
        """
        Default trainer calc after train. Assumes trainer has a 'get_calc' method.
        """
        return self.trainer.get_calc()  
