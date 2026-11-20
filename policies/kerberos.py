import os
import time
from typing import Any, Dict
import dimod
import hybrid

from subQUBO import SubQUBO
from utils.cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dotenv import load_dotenv

from utils.mockSolverOffline import OfflineQPUSolver


class QPUTimeEstimator(hybrid.Runnable):
    """Intercetta i sub-problemi nel ramo QPU e accumula il tempo stimato."""
    def __init__(self, time_accumulator_dict, key_name, num_reads=40, annealing_time=20, **runopts):
        super().__init__(**runopts)
        self.acc_dict = time_accumulator_dict
        self.key_name = key_name
        self.num_reads = num_reads
        self.annealing_time = annealing_time

    def next(self, state, **runopts):
        sub_bqm = getattr(state, 'subproblem', None)
        if sub_bqm and len(sub_bqm.variables) > 0:
            num_qubits = len(sub_bqm.variables)
            #calcola il tempo con solver fittizio
            estimated_us = OfflineQPUSolver.estimate_qpu_access_time(
                num_qubits=num_qubits,
                num_reads=self.num_reads,
                annealing_time=self.annealing_time
            )
            self.acc_dict[self.key_name] += estimated_us
            
        return state

class RaceCounter(hybrid.Runnable):
    """
    Runnable custom compatibile con hybrid.Race e thread asincroni.
    Traccia l'esecuzione incrementando il contatore senza alterare lo State.
    """
    def __init__(self, counter_dict, key_name, **runopts):
        super().__init__(**runopts)
        self.counter_dict = counter_dict
        self.key_name = key_name

    def next(self, state, **runopts):

        self.counter_dict[self.key_name] += 1
        return state

class KerberosPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="kerberos", **kwargs)

    def solve(self, 
              #meta, bqm, seed, global_embedding,is_all_embeddable, qpu_identity, target_graph=None, 
              test_type = "normal") -> Dict[str, Any]:
        lp_filename = self.meta.get("lp_filename", None)
        bestknown_solution = float(self.meta.get("bestknown_solution",None))

        invert_func = self.meta.get("invert_func", None)
                
        cqm = self.meta.get("cqm",None)

        is_maximize = self.meta.get("is_maximize", False)

        mybqm = SubQUBO(self.bqm)
        
        bqm_num_vars = mybqm.num_variables
        bqm_avg_degree = mybqm.average_degree
        bqm_density = mybqm.density

        #dizionario per tracciare le invocazioni esatte di ciascun ramo
        calls = {
            "cpu_tabu_calls": 0,
            "cpu_sa_calls": 0,
            "qpu_calls": 0,
            "estimated_leap_qpu_time_us": 0.0
        }

        real_kerberos = False
        if self.qpu_identity:
            qpu_sampler = DWaveSampler(solver=dict(identity=self.qpu_identity))

            workflow = hybrid.Kerberos(max_iter=self.max_iter,convergence=self.convergence, 
                                                  qpu_reads = 40, qpu_params={'annealing_time': 20}, sa_reads=20,sa_sweeps=1000,tabu_timeout=200,
                                                   max_subproblem_size=min(self.subproblem_size, bqm_num_vars), qpu_sampler=qpu_sampler)
            real_kerberos = True
        else:
            qpu_branch = (
                        hybrid.EnergyImpactDecomposer(
                            size=min(self.subproblem_size, bqm_num_vars),
                            rolling=True,
                            rolling_history=0.3,
                            traversal='bfs'
                        )
                        | hybrid.Const(subsamples=None) #reset subsamples
                        | RaceCounter(calls, "qpu_calls")
                        | QPUTimeEstimator(calls, "estimated_leap_qpu_time_us", num_reads=40, annealing_time=20)
                        | hybrid.SimulatedAnnealingSubproblemSampler(num_reads=10, num_sweeps=500)
                        | hybrid.SplatComposer()
                    )

            # Rami CPU
            tabu_branch = (  
                RaceCounter(calls, "cpu_tabu_calls")
                | hybrid.InterruptableTabuSampler(timeout=500)
            )
            sa_global_branch = (
                RaceCounter(calls, "cpu_sa_calls")
                | hybrid.SimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000)
            )
            workflow = hybrid.Loop(
                hybrid.Race(
                    hybrid.BlockingIdentity(),
                    tabu_branch,
                    sa_global_branch,
                    qpu_branch
                ) | hybrid.ArgMin(),
                max_iter=self.max_iter,
                convergence=self.convergence
            )
            
        start_time = time.perf_counter()
            
        init_sample = dimod.SimulatedAnnealingSampler().sample(self.bqm, num_reads=1).first.sample
            
        init_state = hybrid.State.from_sample(init_sample, self.bqm)
        
        final_state = workflow.run(init_state).result()

        total_wall_clock = time.perf_counter() - start_time

        total_qpu = calls["qpu_calls"]
        total_cpu = calls["cpu_tabu_calls"] + calls["cpu_sa_calls"]
        total_calls = total_qpu + total_cpu

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
            "qpu_calls": total_qpu,
            "leap_qpu_calls": total_qpu if real_kerberos else 0,
            "cpu_calls": total_cpu,
            "qpu_utilization_ratio": total_qpu/total_calls if total_calls > 0 and not real_kerberos else 0.0,
            "real_qpu_utilization_ratio": total_qpu / total_calls if total_calls > 0 and real_kerberos else 0.0,
            
            **cqmResults,
            "estimated_leap_qpu_time_us": calls["estimated_leap_qpu_time_us"],
        }