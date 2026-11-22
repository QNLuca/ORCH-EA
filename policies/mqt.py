import os
import hybrid
import dimod
import time
from typing import Dict, Any
import pickle
import pandas as pd
import numpy as np

from utils.cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy
from policies.baseRouter import BaseRouter

#per sopprimere warning sklearn, tornare indietro di versione romperebbe l'attuale versione di Ocean SDK D-Wave 
import warnings
from sklearn.exceptions import InconsistentVersionWarning

from subQUBO import SubQUBO

from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_TOKEN = os.getenv("DWAVE_API_TOKEN")

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class MQTAORouter(BaseRouter):
    """Router basato sul classificatore ML di MQT-QAO."""
    def __init__(self, qpu_sampler, cpu_sampler, test_type="normal", qpu_identity=None, target_graph=None, chain_strength_prefactor=None, seed=None, real_qpu_path = False, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, test_type=test_type, cpu_sampler=cpu_sampler,chain_strength_prefactor=chain_strength_prefactor, 
                         target_graph=target_graph, seed=seed, qpu_identity=qpu_identity,real_qpu_path=real_qpu_path, **kwargs)


    def extract_qubo_features(self,bqm: dimod.BinaryQuadraticModel) -> pd.DataFrame:
        num_vars = len(bqm.variables)
        linear_coeffs = list(bqm.linear.values())
        num_a = len(linear_coeffs)
        mean_a = float(np.mean(linear_coeffs)) if num_a > 0 else 0.0
        var_a = float(np.var(linear_coeffs)) if num_a > 0 else 0.0

        quad_coeffs = list(bqm.quadratic.values())
        num_b = len(quad_coeffs)
        mean_b = float(np.mean(quad_coeffs)) if num_b > 0 else 0.0
        var_b = float(np.var(quad_coeffs)) if num_b > 0 else 0.0

        all_coeffs = linear_coeffs + quad_coeffs
        mean_all = float(np.mean(all_coeffs)) if len(all_coeffs) > 0 else 0.0
        var_all = float(np.var(all_coeffs)) if len(all_coeffs) > 0 else 0.0

        feature_names = [
            "var qubo", "n 1", "n 2", "avg 1", "avg 2", 
            "var 1", "var 2", "total average", "total variance"
        ]
        features = [[num_vars, num_a, num_b, mean_a, mean_b, var_a, var_b, mean_all, var_all]]
        return pd.DataFrame(features, columns=feature_names)

    def predict_mqt_best_solver(self,bqm: dimod.BinaryQuadraticModel) -> str:
            try:
                #importato modelli dal lavoro di Volpe D. et al, 2024: https://github.com/cda-tum/mqt-qao/tree/main/src/mqt/qao/model/RandomForest
                with open(os.path.join("modelsRF", "model.pkl"), "rb") as f:
                    model = pickle.load(f)
                with open(os.path.join("modelsRF", "Scaler.pkl"), "rb") as f:
                    scaler = pickle.load(f)
                with open(os.path.join("modelsRF", "ScalerKCross.pkl"), "rb") as f:
                    scalerk = pickle.load(f)
    
                features = self.extract_qubo_features(bqm)
                features_scaled = scalerk.transform(features)
                features_scaled = scaler.transform(features_scaled)
    
                prediction = int(model.predict(features_scaled)[0])
                label_map = {0: 'QA', 1: 'QAOA', 2: 'VQE', 3: 'GAS', 4: 'SA'}
                return label_map.get(prediction, "SA")
            except Exception as e:
                print(e)
                return "SA"

    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo = SubQUBO(sub_bqm)

        t_route0 = time.perf_counter()
        solver_scelto = self.predict_mqt_best_solver(sub_bqm)
        t_route = time.perf_counter() - t_route0
        
        is_embeddable = False
        embedding = None

        if self.test_type == "regret":
            #ho bisogno di embedding per calcolare correttamente metrica regret
            print(f"regret test su sottoproblema in corso...")
            t_emb_sub0 = time.perf_counter()
            embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph, self.seed)
            t_emb_sub = time.perf_counter() - t_emb_sub0
        else:
            if solver_scelto == 'QA':
                t_emb_sub0 = time.perf_counter()
                embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph, self.seed)
                t_emb_sub = time.perf_counter() - t_emb_sub0

        route_to_qpu = solver_scelto == 'QA' and is_embeddable

        res_state = self._execute_route(state, route_to_qpu=route_to_qpu, embedding=embedding)
        
        proxy_metrics = res_state.info.get("proxy_metrics", {})

        if proxy_metrics:
            self._compute_proxy_metrics(proxy_metrics)

        with self.lock:
            self.total_routing_time += t_route
            if solver_scelto == 'QA' or self.test_type == "regret":
                self.total_sub_embedding_time += t_emb_sub

        return state.updated(subsamples=res_state)

class MQTQAOPolicy(BasePolicy):
    """
    Policy basata sulla selezione predittiva del solutore tramite Machine Learning (modello Random Forest).

    Lavori a riguardo:
        - Volpe, D., et al. (2024). "A Predictive Approach for Selecting the Best Quantum Solver 
          for an Optimization Problem". IEEE QCE. arXiv:2408.03613.
        - Volpe, D., et al. (2024). "Towards an Automatic Framework for Solving Optimization 
          Problems with Quantum Computers". IEEE QSW. arXiv:2406.12840.
    """
    def __init__(self, **kwargs):
        super().__init__(name="mqt_qao", **kwargs)

    def solve(self, test_type = "normal", real_qpu_path=False) -> Dict[str, Any]:

        chain_strength_prefactor = self.meta.get('chain_strength_prefactor', None)

        mybqm = SubQUBO(self.bqm)

        bqm_num_vars = mybqm.num_variables
        bqm_avg_degree = mybqm.average_degree
        bqm_density = mybqm.density
        
        qpu_target = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = MQTAORouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=self.target_graph,
            chain_strength_prefactor = chain_strength_prefactor,
            seed=self.seed,
            qpu_identity=self.qpu_identity,
            test_type = test_type,
            real_qpu_path=real_qpu_path
        )

        start_time = time.perf_counter()
                
        predicted_solver = router.predict_mqt_best_solver(self.bqm)

        if predicted_solver == "QA" and self.is_all_embeddable:

            if DWAVE_API_TOKEN and real_qpu_path:
                real_qpu = True
            else:
                real_qpu = False

            return self.qpu_direct_branch(real_qpu=real_qpu)

        lp_filename = self.meta.get("lp_filename", None)
        
        bestknown_solution = float(self.meta.get("bestknown_solution",None))

        invert_func = self.meta.get("invert_func", None)
                
        cqm = self.meta.get("cqm",None)

        is_maximize = self.meta.get("is_maximize", False)

        decomposer = hybrid.Unwind(
            hybrid.EnergyImpactDecomposer(
                size=min(self.subproblem_size, bqm_num_vars),
                rolling=True,
                rolling_history=0.3,
                traversal='bfs'
            )
        )

        subproblem_pipeline = (
            decomposer
            | hybrid.Map(router)
            | hybrid.Reduce(hybrid.Lambda(self._merge_substates))
            | hybrid.SplatComposer()
        )

        iteration = hybrid.ParallelBranches(
            hybrid.SimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000),
            hybrid.TabuProblemSampler(),
            subproblem_pipeline
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)
        init_sample = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, self.bqm)

        final_state = main.run(init_state).result()

        final_time = time.perf_counter() - start_time

        total_wall_clock = final_time + self.embedding_time

        qpu_calls = router.leap_qpu_calls + router.fallback_qpu_calls
        total_calls = qpu_calls + router.cpu_calls

        #best_sample = dict(final_state.samples.first.sample)

        cqmResults = {}
        
        if cqm is not None and len(cqm.constraints) > 0:
            all_samples = final_state.samples #sarà solo 1
            cqmResults = CqmUtilities.evaluate_cqm_feasibility(all_samples,invert_func, cqm, is_maximize)
            best_energy = cqmResults['cqm_objective_energy']
        else:
            raw_energy = float(final_state.samples.first.energy)
            best_energy = -raw_energy if is_maximize else raw_energy

        rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
        abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

        all_estimated_qpu = router.estimated_runtime_total

        all_regret_mean_energy_gap = np.mean(router.all_regret_energy_gap) if getattr(router, "all_regret_energy_gap", None) else 0.0

        real_qpu_stats = {
            "qpu_sampling_time": getattr(router, "total_qpu_sampling_time", 0),
            "qpu_anneal_time_per_sample": (np.mean(router.all_qpu_anneal_time_per_sample)) if getattr(router, "all_qpu_anneal_time_per_sample", None) else 0.0,
            "qpu_readout_time_per_sample": (np.mean(router.all_qpu_readout_time_per_sample)) if getattr(router, "all_qpu_readout_time_per_sample", None) else 0.0,
            "qpu_access_time": getattr(router, "total_qpu_access_time", 0.0),
            "qpu_access_overhead_time": getattr(router, "total_qpu_access_overhead_time", 0.0),
            "qpu_programming_time": getattr(router, "total_qpu_programming_time", 0.0),
            "qpu_delay_time_per_sample": (np.mean(router.all_qpu_delay_time_per_sample)) if getattr(router, "all_qpu_delay_time_per_sample", None) else 0.0,
            "total_post_processing_time": getattr(router, "total_total_post_processing_time", 0.0),
            "post_processing_overhead_time": getattr(router, "total_post_processing_overhead_time", 0.0),
        }

        eqi_stats = {
            "eqi_mean": float(np.mean(router.all_eqi)) if getattr(router, "all_eqi", None) else 0.0,
            "max_chain": float(np.mean(router.all_max_chains)) if getattr(router, "all_max_chains", None) else 0.0,
            "mean_chain": float(np.mean(router.all_mean_chains)) if getattr(router, "all_mean_chains", None) else 0.0,
            "median_chain": float(np.mean(router.all_median_chains)) if getattr(router, "all_median_chains", None) else 0.0,
            "physical_nodes": float(np.mean(router.all_physical_nodes)) if getattr(router, "all_physical_nodes", None) else 0.0,
            "expansion": float(np.mean(router.all_expansions)) if getattr(router, "all_expansions", None) else 0.0,
            "long_chain_fraction": float(np.mean(router.all_long_chain_fractions)) if getattr(router, "all_long_chain_fractions", None) else 0.0,
        }
        
        avg_chain_break = float(np.mean(router.chain_break_fractions)) if router.chain_break_fractions else 0.0

        return {
            "instance_id": lp_filename,
            "seed": self.seed,
            "num_vars": bqm_num_vars,
            "avg_degree": bqm_avg_degree,
            "density": bqm_density,
            "is_maximization_objective": is_maximize,
            "policy": self.name,
            "best_energy": best_energy,
            "qplib_reference": bestknown_solution,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "wall_clock_time": total_wall_clock,
            "time_global_embedding": self.embedding_time,
            "qpu_calls": qpu_calls,
            "leap_qpu_calls": router.leap_qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": qpu_calls/total_calls if total_calls > 0 else 0.0,
            "real_qpu_utilization_ratio": router.leap_qpu_calls / total_calls if total_calls > 0 else 0.0,
            "tp": router.tp, "fp": router.fp, "tn": router.tn, "fn": router.fn,
            "regret_mean_energy_gap": all_regret_mean_energy_gap,
            "time_subqubo_embedding": router.total_sub_embedding_time,
            "time_total_embedding": self.embedding_time + router.total_sub_embedding_time,
            "time_routing_total": router.total_routing_time,
            "decomposed": True,

            "chain_strength_prefactor": chain_strength_prefactor if chain_strength_prefactor is not None else 1.414,

            "chain_break_fraction": avg_chain_break,
            "physical_build_time": router.total_build_time,
            "solve_time": router.total_solve_time,
            "unembed_time": router.total_unembed_time,

            **real_qpu_stats,
            **eqi_stats,
            **cqmResults,
            "estimated_leap_qpu_time_us": all_estimated_qpu,        
        }