import numpy as np
import pandas as pd

from optimizer.swarm import Swarm
import utils


class BasePSOptimizer:
    def __init__(self, swarms, inference_model, fitness_name, opt_col):
        self.infer_model = inference_model
        self.swarms = swarms
        self.best_solutions = pd.DataFrame(columns=opt_col)
        self.best_fitness_history = pd.DataFrame(columns=['step'] + opt_col)
        self.fitness_name = fitness_name
        self.best_step_mof_history = pd.DataFrame(columns=opt_col)
        self.step_fitness_history = pd.DataFrame(columns=['step', 'max_fitness', 'min_fitness', 'mean_fitness'])

    def update_fitness(self, swarm):
        fitness = np.ones(swarm.num_part)
        uptake_scores = [swarm.mof[index][self.fitness_name] for index in range(swarm.num_part)]
        smiles = [swarm.mof[index]['branch_smiles'] for index in range(swarm.num_part)]

        smiles_scores = []
        for smile in smiles:
            smiles_scores.append(utils.capacity_score_smiles(smile))
        fitness = uptake_scores

        for index, smiles_score in enumerate(smiles_scores):
            if not smiles_score:
                fitness[index] = 0
        swarm.update_fitness(fitness)
        return swarm

    def _next_step_and_evaluate(self, swarm):
        swarm.next_step()
        mof_building = self.infer_model.mof_z_to_mof_building(swarm.x)
        mof_y = self.infer_model.mof_z_to_mof_y(swarm.x)
        mof_dict = self.infer_model.mof_building_and_mof_y_to_mof_dict(mof_building, mof_y)
        swarm.mof = mof_dict
        self.update_fitness(swarm)
        return swarm

    def _update_best_solutions(self, num_track):
        df_list = []
        for swarm in self.swarms:
            for index, mof in enumerate(swarm.mof):
                mof_dict = {
                    'fitness': swarm.fitness[index],
                    'organic_core': mof['organic_core'],
                    'metal_node': mof['metal_node'],
                    'topology': mof['topology'],
                    'branch_smiles': mof['branch_smiles'],
                    'lcd': mof['lcd'],
                    'pld': mof['pld'],
                    'density': mof['density'],
                    'agsa': mof['agsa'],
                    'co2n2_co2_mol_kg': mof['co2n2_co2_mol_kg'],
                    'co2n2_n2_mol_kg': mof['co2n2_n2_mol_kg'],
                    'co2ch4_co2_mol_kg': mof['co2ch4_co2_mol_kg'],
                    'co2ch4_ch4_mol_kg': mof['co2ch4_ch4_mol_kg']
                }
                df_list.append(mof_dict)
        new_df = pd.DataFrame(df_list)
        if self.best_solutions.empty:
            best_solutions = new_df
        else:
            best_solutions = pd.concat([self.best_solutions, new_df], sort=False, ignore_index=True)
        best_solutions = best_solutions.sort_values("fitness", ascending=False).reset_index(drop=True)
        self.best_solutions = best_solutions.iloc[:num_track]
        best_solutions_max = self.best_solutions.fitness.max()
        best_solutions_min = self.best_solutions.fitness.min()
        best_solutions_mean = self.best_solutions.fitness.mean()
        return best_solutions_max, best_solutions_min, best_solutions_mean

    def _update_best_fitness_history(self, step):
        df_list = []
        for swarm in self.swarms:
            best_mof_dict = {
                'step': step,
                'fitness': swarm.swarm_best_fitness,
                'organic_core': swarm.best_mof['organic_core'],
                'metal_node': swarm.best_mof['metal_node'],
                'topology': swarm.best_mof['topology'],
                'branch_smiles': swarm.best_mof['branch_smiles'],
                'lcd': swarm.best_mof['lcd'],
                'pld': swarm.best_mof['pld'],
                'density': swarm.best_mof['density'],
                'agsa': swarm.best_mof['agsa'],
                'co2n2_co2_mol_kg': swarm.best_mof['co2n2_co2_mol_kg'],
                'co2n2_n2_mol_kg': swarm.best_mof['co2n2_n2_mol_kg'],
                'co2ch4_co2_mol_kg': swarm.best_mof['co2ch4_co2_mol_kg'],
                'co2ch4_ch4_mol_kg': swarm.best_mof['co2ch4_ch4_mol_kg']
            }
            df_list.append(best_mof_dict)
        df = pd.DataFrame(df_list)
        if self.best_fitness_history.empty:
            self.best_fitness_history = df
        else:
            self.best_fitness_history = pd.concat([self.best_fitness_history, df], sort=False, ignore_index=True)

    def run(self, num_steps, num_track=10):
        for swarm in self.swarms:
            self.update_fitness(swarm)
        for step in range(num_steps):
            for swarm in self.swarms:
                self._next_step_and_evaluate(swarm)
            self._update_best_fitness_history(step)
            max_fitness, min_fitness, mean_fitness = self._update_best_solutions(num_track)
            if self.step_fitness_history.empty:
                self.step_fitness_history = pd.DataFrame([(step, max_fitness, min_fitness, mean_fitness)],
                                                         columns=['step', 'max_fitness', 'min_fitness', 'mean_fitness'])
            else:
                self.step_fitness_history = pd.concat(
                    [self.step_fitness_history, pd.DataFrame([(step, max_fitness, min_fitness, mean_fitness)],
                                                             columns=['step', 'max_fitness',
                                                                      'min_fitness', 'mean_fitness'])])
        for swarm in self.swarms:
            if self.best_step_mof_history.empty:
                self.best_step_mof_history = pd.DataFrame(swarm.history_swarm_best_step_mof)
            else:
                self.best_step_mof_history = pd.concat(
                    (self.best_step_mof_history, pd.DataFrame(swarm.history_swarm_best_step_mof)))
        return self.swarms

    @classmethod
    def from_query(cls, init_building_list, num_part, num_swarms, inference_model, fitness_name, opt_col,
                   phi1=2., phi2=2., phi3=2., x_min=-5.,
                   x_max=5., v_min=-0.3, v_max=0.3):

        idxs = np.random.randint(0, len(init_building_list), size=num_part)
        init_building_list = [init_building_list[i] for i in idxs]

        mof_tensor = inference_model.mof_building_to_mof_tensor(init_building_list)
        mof_y = inference_model.mof_tensor_to_mof_y(mof_tensor)
        mof_dict = inference_model.mof_building_and_mof_y_to_mof_dict(init_building_list, mof_y)
        mof_z = inference_model.mof_tensor_to_mof_z(mof_tensor)

        swarms = [
            Swarm.from_query(
                init_mof=mof_dict,
                init_emb=mof_z,
                num_part=num_part,
                v_min=v_min,
                v_max=v_max,
                x_min=x_min,
                x_max=x_max,
                phi1=phi1,
                phi2=phi2,
                phi3=phi3) for _ in range(num_swarms)]
        return cls(swarms, inference_model, fitness_name, opt_col)
