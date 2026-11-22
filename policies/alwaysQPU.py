import os

import hybrid
import dimod
import time
from typing import Dict, Any
import numpy as np

from utils.cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy
from policies.baseRouter import BaseRouter
from subQUBO import SubQUBO


from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_TOKEN = os.getenv("DWAVE_API_TOKEN")

class AlwaysQPUEmbeddableRouter(BaseRouter):
    """Invia a QPU ogni sub-problema per cui si riesce a calcolare un embedding valido."""
    def __init__(self, qpu_sampler, cpu_sampler, test_type ="normal", qpu_identity = None, target_graph=None, chain_strength_prefactor=None,seed = None, real_qpu_path=False, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler, test_type=test_type, chain_strength_prefactor=chain_strength_prefactor,
                         seed=seed, target_graph=target_graph, qpu_identity=qpu_identity, real_qpu_path=real_qpu_path, **kwargs)


    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo = SubQUBO(sub_bqm)
        
        t_emb_sub0 = time.perf_counter()
        embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph, self.seed)
        t_emb_sub = time.perf_counter() - t_emb_sub0

        res_state = self._execute_route(state, route_to_qpu=is_embeddable, embedding=embedding)

        proxy_metrics = res_state.info.get("proxy_metrics", {})

        if proxy_metrics:
            self._compute_proxy_metrics(proxy_metrics)

        with self.lock:
            self.total_sub_embedding_time += t_emb_sub
            #qui routing time è logicamente 0 dato che calcolo separatamente embedding time sottoproblemi

        return state.updated(subsamples=res_state)

class AlwaysQPUEmbeddablePolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="always_qpu", **kwargs)

    def solve(self,test_type = "normal", real_qpu_path = False) -> Dict[str, Any]:

        start_time = time.perf_counter()

        
        if self.is_all_embeddable:

            if DWAVE_API_TOKEN and real_qpu_path: #and nella richiesta flag "real_qpu"
                real_qpu = True
            else:
                real_qpu = False

            return self.qpu_direct_branch(real_qpu=real_qpu)


        lp_filename = self.meta.get("lp_filename", None)
        bestknown_solution = float(self.meta.get("bestknown_solution",None))

        invert_func = self.meta.get("invert_func", None)
                
        cqm = self.meta.get("cqm",None)

        is_maximize = self.meta.get("is_maximize", False)

        chain_strength_prefactor = self.meta.get('chain_strength_prefactor')

        mybqm = SubQUBO(self.bqm)
        
        bqm_num_vars = mybqm.num_variables
        bqm_avg_degree = mybqm.average_degree
        bqm_density = mybqm.density

        qpu_target = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = AlwaysQPUEmbeddableRouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=self.target_graph,
            chain_strength_prefactor = chain_strength_prefactor,
            seed=self.seed,
            qpu_identity= self.qpu_identity,
            test_type = test_type,
            real_qpu_path = real_qpu_path
        )

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

        #nessun apporto ibrido

        main = hybrid.Loop(subproblem_pipeline, max_iter=self.max_iter, convergence=self.convergence)

        init_sample = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, self.bqm)

        final_state = main.run(init_state).result()

        final_time = time.perf_counter() - start_time

        total_wall_clock = final_time + self.embedding_time

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

        qpu_calls = router.leap_qpu_calls + router.fallback_qpu_calls
        total_calls = qpu_calls + router.cpu_calls

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
            "qpu_utilization_ratio": qpu_calls / total_calls if total_calls > 0 else 0.0,
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