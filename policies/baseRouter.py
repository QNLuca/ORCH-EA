import random
import threading
import time
import hybrid
from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
import dwave.graphs as dnx

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.cloud.exceptions import SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout
import numpy as np

from subQUBO import SubQUBO
from utils.leapBudget import can_afford, charge
from utils.mockSolverOffline import OfflineQPUSolver

class BaseRouter(hybrid.Runnable):
    """Router base che gestisce le metriche e l'esecuzione su QPU/CPU."""
    def __init__(self, qpu_sampler, cpu_sampler, test_type= "normal", seed=None,qpu_identity=None, target_graph=None, chain_strength_prefactor = None, real_qpu_path = False,**runopts):
        super().__init__(**runopts)
        self.qpu_sampler = qpu_sampler
        self.cpu_sampler = cpu_sampler
        self.leap_qpu_calls = 0
        self.fallback_qpu_calls = 0
        self.cpu_calls = 0
        self.chain_strength_prefactor = chain_strength_prefactor
        self.real_qpu_path = real_qpu_path

        self.test_type=test_type

        self.target_graph = target_graph
        self.seed = seed
        self.qpu_identity = qpu_identity

        self.total_build_time = 0.0
        self.total_solve_time = 0.0
        self.total_unembed_time = 0.0
        self.total_sub_embedding_time = 0.0
        self.total_routing_time = 0.0
        self.chain_break_fractions = []
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.tn = 0  # True Negative
        self.fn = 0  # False Negative

        self.all_eqi = []
        self.all_max_chains = []
        self.all_mean_chains = []
        self.all_median_chains = []
        self.all_long_chain_fractions = []
        self.all_expansions = []
        self.all_physical_nodes = []
        self.estimated_runtime_total = 0

        self.total_qpu_sampling_time = 0.0
        self.all_qpu_anneal_time_per_sample = []
        self.all_qpu_readout_time_per_sample = []
        self.total_qpu_access_time = 0.0
        self.total_qpu_access_overhead_time = 0.0
        self.total_qpu_programming_time = 0.0
        self.all_qpu_delay_time_per_sample = []
        self.total_total_post_processing_time = 0.0
        self.total_post_processing_overhead_time = 0.0

        self.all_regret_energy_gap = []

        self.leap_qpu_sampler = None

        self.lock = threading.Lock()

    def _compute_proxy_metrics(self, proxy_metrics: dict):
                
        with self.lock:
            self.total_solve_time += float(proxy_metrics["solve_time"]) #sia qpu che cpu
            from_qpu = proxy_metrics.get("from_qpu", False)
            real_qpu = proxy_metrics.get("real_qpu", False)
            #good_embedding = proxy_metrics.get("good_embedding", False)
            if "chain_break_fraction" in proxy_metrics:
                self.chain_break_fractions.append(proxy_metrics["chain_break_fraction"])
            if "eqi" in proxy_metrics:
                self.all_eqi.append(proxy_metrics["eqi"])
                self.all_max_chains.append(proxy_metrics["max_chain"])
                self.all_mean_chains.append(proxy_metrics["mean_chain"])
                self.all_median_chains.append(proxy_metrics["median_chain"])
                self.all_long_chain_fractions.append(proxy_metrics["long_chain_fraction"])
                self.all_expansions.append(proxy_metrics["expansion"]) # == mean_chain
                self.all_physical_nodes.append(proxy_metrics["physical_nodes"])
            if from_qpu and not real_qpu:
                #statistiche raccolte solo su ramo quantistico simulato
                self.total_build_time += float(proxy_metrics["physical_build_time"])
                self.total_unembed_time += float(proxy_metrics["unembed_time"])
                self.estimated_runtime_total += float(proxy_metrics["estimated_leap_qpu_time_us"])
            if from_qpu and real_qpu:
                self.total_qpu_sampling_time += float(proxy_metrics["qpu_sampling_time"])
                self.all_qpu_anneal_time_per_sample.append(proxy_metrics["qpu_anneal_time_per_sample"])
                self.all_qpu_readout_time_per_sample.append(proxy_metrics["qpu_readout_time_per_sample"])
                self.total_qpu_access_time += float(proxy_metrics["qpu_access_time"])
                self.total_qpu_access_overhead_time += float(proxy_metrics["qpu_access_overhead_time"])
                self.total_qpu_programming_time += float(proxy_metrics["qpu_programming_time"])
                self.all_qpu_delay_time_per_sample.append(proxy_metrics["qpu_delay_time_per_sample"])
                self.total_total_post_processing_time += float(proxy_metrics["total_post_processing_time"])
                self.total_post_processing_overhead_time += float(proxy_metrics["post_processing_overhead_time"])
                self.estimated_runtime_total += float(proxy_metrics["estimated_leap_qpu_time_us"])

            if self.test_type == "regret":
                regret = proxy_metrics.get("regret",False)
                self.all_regret_energy_gap.append(proxy_metrics["regret_energy_gap"])

                if not regret and from_qpu:
                    self.tp += 1
                elif regret and from_qpu:
                    self.fp += 1
                elif not regret and not from_qpu:
                    self.tn += 1
                elif regret and not from_qpu:
                    self.fn += 1

    def _get_qpu_sampler(self, qpu_identity=None):

        #riusa se già aperto con la stessa identity
        if self.leap_qpu_sampler is not None:
            if qpu_identity is None or self.qpu_identity == qpu_identity:
                return self.leap_qpu_sampler
            #identità diversa: chiudi e ricrea
            try:
                self.leap_qpu_sampler.client.close()
            except Exception:
                pass
            self.leap_qpu_sampler = None

        if qpu_identity:
            self.leap_qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
        else:
            self.leap_qpu_sampler = DWaveSampler(solver={"topology__type": "zephyr"})
            self.qpu_identity = self.qpu_sampler.solver.identity

        return self.leap_qpu_sampler

            

    def _qpu_branch(self, state, embedding):

        qpu_state_clean = state.updated(subsamples=None) #resetta subsamples per evitare errori di mismatch: https://support.dwavesys.com/hc/en-us/community/posts/360037074933-Error-message-when-trying-to-solve-sub-problems

        qpu_state = qpu_state_clean.subproblem

        if self.chain_strength_prefactor is not None:
            chain_strength = uniform_torque_compensation(
                qpu_state, embedding=embedding, prefactor=self.chain_strength_prefactor
            )
        else:
            chain_strength = uniform_torque_compensation(
                qpu_state, embedding=embedding, prefactor=1.414
            )

        successLeap = False
        est_us = None

        if self.qpu_identity and self.real_qpu_path:
            #sono per forza in un benchamrk e dunque se ho API ho anche qpu_identity
            with self.lock:
                #gestione thread-safe dell'avvio del sampler
                qpu_sampler = self._get_qpu_sampler(self.qpu_identity)

            num_phys = len({q for chain in embedding.values() for q in chain})
            est_us = int(qpu_sampler.solver.estimate_qpu_access_time(
                 num_qubits=num_phys,
                 num_reads=40,
                 annealing_time=20,
             ))

            
            if not can_afford(est_us):
               successLeap = False  #va al fallback EAP
            else:
                fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=embedding)

                max_retries = 3
                retry_delay = 1.0
                res_sampleset = None

                for attempt in range(1, max_retries + 1):
                    try:
                        t_start = time.perf_counter()

                        res_sampleset = fixed_composite.sample(
                            qpu_state,
                            chain_strength=chain_strength,
                            num_reads=40,
                            annealing_time = 20
                        )

                        res_sampleset.resolve()

                        t_end = time.perf_counter()
                        wall_service_time = t_end - t_start
                        successLeap = True

                        #æggiorno conteggio calls leap
                        self.leap_qpu_calls += 1
                        break

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

            #estrazione di timing
            if successLeap and res_sampleset is not None:
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

                if "chain_break_fraction" in res_sampleset.record.dtype.names:
                    cbf = res_sampleset.record.chain_break_fraction
                    weights = res_sampleset.record.num_occurrences
                    avg_chain_break = float(np.average(cbf, weights=weights)) if len(cbf) > 0 else 0.0
                else:
                    avg_chain_break = 0.0

                emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, qpu_state, avg_chain_break)

                extra_metrics = {
                    "from_qpu": True,
                    "real_qpu": True,
                    "chain_break_fraction": avg_chain_break,
                    "solve_time": wall_service_time,
                    "estimated_leap_qpu_time_us": est_us,
                    **qpu_full_timing,
                    **emb_metrics
                }

        if not self.qpu_identity or not successLeap:              
            t_build_0 = time.perf_counter()  
            target_bqm = embed_bqm(
                qpu_state, 
                embedding=embedding, 
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

            end_state = self.qpu_sampler.run(init_state).result()

            physical_sampleset = end_state.samples
            t_solve_total = time.perf_counter() - t_start_solve

            t_unembed_0 = time.perf_counter()
            res_sampleset = unembed_sampleset(physical_sampleset, embedding, qpu_state, chain_break_method=majority_vote)
            t_unembed = time.perf_counter() - t_unembed_0
            
            #æggiorno conteggio calls fallback
            self.fallback_qpu_calls += 1

            if "chain_break_fraction" in res_sampleset.record.dtype.names:
                cbf = res_sampleset.record.chain_break_fraction
                weights = res_sampleset.record.num_occurrences
                avg_chain_break = float(np.average(cbf, weights=weights)) if len(cbf) > 0 else 0.0
            else:
                chain_break_fraction = list(chain_break_frequency(physical_sampleset, embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.                       
                avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

            emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, qpu_state, avg_chain_break)

            extra_metrics = {
                            "from_qpu": True,
                            "real_qpu": False,
                            "physical_build_time": t_build,
                            "unembed_time": t_unembed,
                            "solve_time": t_solve_total,
                            "estimated_leap_qpu_time_us": estimated_runtime_us,
                            "chain_break_fraction": avg_chain_break,
                            **emb_metrics
                        }
            
        return res_sampleset, extra_metrics

    def _cpu_branch(self, state, embedding=None):

        clean_state = state.updated(subsamples=None)

        t_solve_0 = time.perf_counter()
        final_state = self.cpu_sampler.run(clean_state).result()
        t_solve = time.perf_counter() - t_solve_0

        res_sampleset = final_state.subsamples

        self.cpu_calls += 1

        if embedding:
             emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, state.subproblem, None)
        else:
            emb_metrics = {
                "has_embedding": False,
                "good_embedding": False
            }

        extra_metrics = {
            "from_qpu": False,
            "solve_time": t_solve,
            **emb_metrics
        }

        return res_sampleset, extra_metrics


    def _execute_route(self, state, route_to_qpu: bool, embedding=None):

        if self.test_type == "regret":
            #calcolo sia ramo QPU che CPU e calcolo regret
            #if route_to_qpu:
            #    with self.lock:
            #        self.qpu_calls += 1
            #else:
            #    with self.lock:
            #        self.cpu_calls +=1

            best_energy_qpu = None
            res_sampleset_qpu = None
            if embedding is not None:
                #senza embedding non posso passarlo al ramo QPU
                res_sampleset_qpu, extra_metrics_qpu = self._qpu_branch(state,embedding)
                best_energy_qpu = res_sampleset_qpu.first.energy

            res_sampleset_cpu, extra_metrics_cpu = self._cpu_branch(state,embedding)
            best_energy_cpu = res_sampleset_cpu.first.energy

            if best_energy_qpu is not None:
                regret = (best_energy_qpu>best_energy_cpu and route_to_qpu) or (best_energy_qpu<best_energy_cpu and not route_to_qpu)
            else:
                #se non ho potuto eseguire ramo QPU, no regret
                regret = False

            res_sampleset = res_sampleset_qpu if route_to_qpu else res_sampleset_cpu
            extra_metrics = extra_metrics_qpu if route_to_qpu else extra_metrics_cpu

            extra_metrics["regret"] = regret
            if regret:
                #valore di energy gap è positivo quando scelta fatta è peggiore
                chosen_energy = best_energy_qpu if route_to_qpu else best_energy_cpu
                optimal_energy = best_energy_cpu if route_to_qpu else best_energy_qpu

                extra_metrics["regret_energy_gap"] = chosen_energy - optimal_energy
            else:
                extra_metrics["regret_energy_gap"] = 0.0

        elif route_to_qpu:

            res_sampleset, extra_metrics = self._qpu_branch(state,embedding)

        else:

            res_sampleset, extra_metrics = self._cpu_branch(state,embedding)


        if hasattr(res_sampleset, 'info') and isinstance(res_sampleset.info, dict):
            res_sampleset.info["proxy_metrics"] = extra_metrics

        return res_sampleset