import random
import time

import hybrid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.cloud.exceptions import SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from subQUBO import SubQUBO
from utils.cqmUtilities import CqmUtilities
from utils.leapBudget import can_afford, charge
from utils.mockSolverOffline import OfflineQPUSolver

class BasePolicy(ABC):
    """Classe base astratta per tutte le policy di risoluzione."""
    
    def __init__(self, name: str, meta, bqm, is_all_embeddable,target_graph,seed,qpu_identity, global_embedding = None, embedding_time: Optional[float] = None, subproblem_size: int = 40, max_iter: int = 3, convergence: int = 3):
        self.name = name
        self.subproblem_size = subproblem_size
        self.max_iter = max_iter
        self.convergence = convergence
        self.embedding_time = embedding_time
        self.global_embedding = global_embedding
        self.meta = meta
        self.bqm = bqm
        self.is_all_embeddable = is_all_embeddable
        self.target_graph = target_graph
        self.seed = seed
        self.qpu_identity = qpu_identity

    @abstractmethod
    def solve(
        self, 
        #bqm: dimod.BinaryQuadraticModel, 
        #global_embedding: Dict, 
        #is_all_embeddable: bool, 
        #seed,
        #qpu_identity = None,
        #target_graph = None,
        test_type = "normal"
    ) -> Dict[str, Any]:
        """Metodo astratto da implementare in ogni policy specifica."""
        pass

    def qpu_direct_branch(self,real_qpu):
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

            qpu_target_global = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)

            start_time = time.perf_counter()

            if self.meta is not None and chain_strength_prefactor is not None:
                pass
            else:
                chain_strength_prefactor = 1.414

            chain_strength = uniform_torque_compensation(
                self.bqm, embedding=self.global_embedding, prefactor=chain_strength_prefactor
            )

            successLeap = False
            qpu_sampler = None
            est_us = None

            try:
                if real_qpu:

                    qpu_sampler = DWaveSampler(solver=dict(identity=self.qpu_identity))

                    physical_nodes = len({q for chain in self.global_embedding.values() for q in chain})
                    est_us = int(qpu_sampler.solver.estimate_qpu_access_time(
                        num_qubits=physical_nodes,
                        num_reads=40,
                        annealing_time=20,
                    ))

                    if not can_afford(est_us):
                        successLeap = False  #va al fallback EAP
                    else:
                        fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=self.global_embedding)

                        t_start = time.perf_counter()

                        max_retries = 3
                        retry_delay = 1.0
                        res_sampleset = None

                        for attempt in range(1, max_retries + 1):
                            try:
                                t_start = time.perf_counter()   
                                res_sampleset = fixed_composite.sample(
                                    self.bqm,
                                    chain_strength=chain_strength,
                                    num_reads=40,
                                    annealing_time=20
                                )

                                res_sampleset.resolve()

                                t_end = time.perf_counter()
                                wall_service_time = t_end - t_start

                                final_time = t_end - start_time
                                successLeap = True
                                break  #successo, esco dal ciclo
                            
                            except (SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout) as e:
                                print(f"Tentativo {attempt}/{max_retries} fallito sulla QPU: {type(e).__name__}: {e}")
                                if attempt == max_retries:
                                    print(f"QPU non disponibile dopo {max_retries} tentativi. "
                                                    f"Procedo con il fallback locale.")
                                    #esco dal ciclo
                                    break
                                #attesa variabile prima del prossimo tentativo
                                sleep_time = retry_delay + random.uniform(0, 0.5)
                                time.sleep(sleep_time)
                                retry_delay *= 2
                            except Exception as e:
                                #errori non temporanei nessun altro retry
                                print(f"Errore non retryable su QPU: {type(e).__name__}: {e}")
                                break

                    if successLeap and res_sampleset is not None:
                        
                        if "chain_break_fraction" in res_sampleset.record.dtype.names:
                            cbf = res_sampleset.record.chain_break_fraction
                            weights = res_sampleset.record.num_occurrences
                            avg_chain_break = float(np.average(cbf, weights=weights)) if len(cbf) > 0 else 0.0
                        else:
                            avg_chain_break = 0.0
 

                        timing_info = res_sampleset.info.get('timing', {})
                        
                        #Raccolti tutti metadati temporali
                        qpu_full_timing = {
                            "qpu_sampling_time": timing_info.get('qpu_sampling_time'),
                            "qpu_anneal_time_per_sample": timing_info.get('qpu_anneal_time_per_sample'),
                            "qpu_readout_time_per_sample": timing_info.get('qpu_readout_time_per_sample'),
                            "qpu_access_time": timing_info.get('qpu_access_time'),
                            "qpu_access_overhead_time": timing_info.get('qpu_access_overhead_time'),
                            "qpu_programming_time": timing_info.get('qpu_programming_time'),
                            "qpu_delay_time_per_sample": timing_info.get('qpu_delay_time_per_sample'),
                            "total_post_processing_time": timing_info.get('total_post_processing_time'),
                            "post_processing_overhead_time": timing_info.get('post_processing_overhead_time'),
                        }                

                        actual_qat = qpu_full_timing['qpu_access_time']
                        charge(actual_qat)

                        extra_metrics = {
                            "solve_time": wall_service_time,
                            "estimated_leap_qpu_time_us": est_us,
                             **qpu_full_timing,
                        }        

                if not real_qpu or not successLeap:               
                    t_build_0 = time.perf_counter()         
                    target_bqm = embed_bqm(
                        self.bqm, 
                        embedding=self.global_embedding, 
                        target_adjacency=self.target_graph,
                        chain_strength=chain_strength
                    )
                    t_build = time.perf_counter() - t_build_0

                    ################# PER TEST LOCALE QPU_ACCESS_TIME #####################

                    if est_us is not None:
                        estimated_runtime_us = est_us
                    else:
                       estimated_runtime_us = OfflineQPUSolver.estimate_qpu_access_time(
                           num_qubits=len(target_bqm.variables), 
                           num_reads=40, 
                           annealing_time=20
                       )

                    ################################################################

                    t_start_solve = time.perf_counter()
                    init_state = hybrid.State.from_problem(target_bqm)

                    end_state = qpu_target_global.run(init_state).result()

                    physical_sampleset = end_state.samples
                    t_solve_total = time.perf_counter() - t_start_solve

                    t_unembed_0 = time.perf_counter()
                    res_sampleset = unembed_sampleset(physical_sampleset, self.global_embedding, self.bqm, chain_break_method=majority_vote)
                    t_unembed = time.perf_counter() - t_unembed_0

                    final_time = time.perf_counter() - start_time

                    if "chain_break_fraction" in res_sampleset.record.dtype.names:
                        cbf = res_sampleset.record.chain_break_fraction
                        weights = res_sampleset.record.num_occurrences
                        avg_chain_break = float(np.average(cbf, weights=weights)) if len(cbf) > 0 else 0.0
                    else:
                        chain_break_fraction = list(chain_break_frequency(physical_sampleset, self.global_embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.                       
                        avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

                    #non uso qpu in questo ramo, ritorno valori a 0
                    qpu_full_timing = {
                            "qpu_sampling_time": 0.0,
                            "qpu_anneal_time_per_sample": 0.0,
                            "qpu_readout_time_per_sample": 0.0,
                            "qpu_access_time": 0.0,
                            "qpu_access_overhead_time": 0.0,
                            "qpu_programming_time": 0.0,
                            "qpu_delay_time_per_sample": 0.0,
                            "total_post_processing_time": 0.0,
                            "post_processing_overhead_time": 0.0,                   
                    }

                    extra_metrics = {
                        "physical_build_time": t_build,
                        "unembed_time": t_unembed,
                        "solve_time": t_solve_total,
                        "estimated_leap_qpu_time_us": estimated_runtime_us,
                        **qpu_full_timing
                    }
            finally:
                if qpu_sampler is not None:
                    try:
                        qpu_sampler.client.close()
                    except Exception:
                        pass

            total_wall_clock = final_time + self.embedding_time

            #best_sample = dict(res_sampleset.first.sample)
            
            emb_metrics = SubQUBO.compute_advanced_embedding_metrics(self.global_embedding,self.bqm,avg_chain_break)

            #good_embedding = bool(emb_metrics["good_embedding"])

            eqi = emb_metrics["eqi"]
            max_chain = float(emb_metrics["max_chain"])
            median_chain = float(emb_metrics["median_chain"])
            mean_chain = float(emb_metrics["mean_chain"])
            long_chain_fraction = float(emb_metrics["long_chain_fraction"])
            physical_nodes = float(emb_metrics["physical_nodes"])
            expansion = float(emb_metrics["expansion"])


            cqmResults = {}
        
            if cqm is not None and len(cqm.constraints) > 0:
                all_samples = res_sampleset
                cqmResults = CqmUtilities.evaluate_cqm_feasibility(all_samples,invert_func, cqm, is_maximize)
                best_energy = cqmResults['cqm_objective_energy']
            else:
                raw_energy = float(res_sampleset.first.energy)
                best_energy = -raw_energy if is_maximize else raw_energy

            rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
            abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None


            return {
                "instance_id": lp_filename,
                "seed": self.seed,
                "num_vars": bqm_num_vars,
                "avg_degree": bqm_avg_degree,
                "density": bqm_density,
                "policy": self.name,
                "is_maximization_objective": is_maximize,
                "best_energy": best_energy,
                "qplib_reference": bestknown_solution,
                "abs_gap": abs_gap,
                "rel_gap": rel_gap,
                "wall_clock_time": total_wall_clock,
                "time_global_embedding": self.embedding_time,
                "qpu_calls": 1,
                "leap_qpu_calls": 1 if successLeap else 0,
                "cpu_calls": 0,
                "qpu_utilization_ratio": 1.0,
                "real_qpu_utilization_ratio": 1.0 if successLeap else 0.0,

                "chain_break_fraction": avg_chain_break,
                "decomposed": False,

                "chain_strength_prefactor": chain_strength_prefactor if chain_strength_prefactor is not None else 1.414,

                "eqi_mean": eqi,
                "max_chain": max_chain,
                "mean_chain": mean_chain,
                "median_chain": median_chain,
                "physical_nodes": physical_nodes,
                "expansion": expansion,
                "long_chain_fraction": long_chain_fraction,

                **extra_metrics,
                **cqmResults,
            }


    @staticmethod
    def _merge_substates(_, substates, **kwargs):
        a, b = substates
        return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))