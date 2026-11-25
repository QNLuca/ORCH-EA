import os
import time
import numpy as np
import redis
from rq import Worker, Queue, Connection, get_current_job, Retry
from rq.job import Job
from rq.timeouts import JobTimeoutException
import dimod
import hybrid
import dwave.graphs as dnx
import pickle
from utils.mockSolverOffline import OfflineQPUSolver 

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.cloud import Client
from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.cloud.exceptions import SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout
import random

from subQUBO import SubQUBO
from utils.leapBudget import can_afford, charge
from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_TOKEN = os.getenv("DWAVE_API_TOKEN")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


def _build_qpu_metrics(res_sampleset: dimod.SampleSet, embedding, sub_bqm: dimod.BinaryQuadraticModel, wall_service_time: float) -> dict:
    """Helper condiviso tra esecuzione normale e recovery per estrarre le metriche QPU."""
    timing_info = res_sampleset.info.get('timing', {})

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

    chain_break_fraction = res_sampleset.record.chain_break_fraction
    avg_chain_break_fraction = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0
    emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, avg_chain_break_fraction)

    return {
        "from_qpu": True,
        "real_qpu": True,
        "qpu_leap_failure": False,
        "solve_time": wall_service_time,
        "chain_break_fraction": avg_chain_break_fraction,
        **qpu_full_timing,
        **emb_metrics,
    }

def manage_qpu_fail(job, connection, type, value, traceback):
    """
    Hook on_failure di RQ per JobTimeoutException, valido solo su sistema Linux.
    Se il problema era già stato caricato su Leap, accoda il recupero a costo zero.
    """
    if issubclass(type, JobTimeoutException) and job.meta.get("dwave_problem_id"):
        print(f"[HOOK ON_FAILURE] Job {job.id} in timeout. Accodo recovery per problem_id={job.meta['dwave_problem_id']}")
        q_recovery = Queue("recovery_queue", connection=connection)
        q_recovery.enqueue(
            recover_sample_from_leap,
            args=(job.id,),
            job_timeout=120,
            retry=Retry(max=3, interval=[15, 30, 60]),
        )


def recover_sample_from_leap(failed_job_id: str):
    """
    Scarica il risultato completato da Leap dopo un timeout RQ,
    ricostruisce unembed + proxy_metrics e pubblica su job_done:{failed_job_id}.
    """
    current_job = get_current_job()
    redis_conn = current_job.connection
    original_job = Job.fetch(failed_job_id, connection=redis_conn)

    problem_id = original_job.meta.get("dwave_problem_id")
    if not problem_id:
        return {"status": "NO_ID"}

    #recupero sub_bqm ed embedding dagli argomenti salvati nel job originale
    sub_bqm = original_job.args[0]
    embedding = original_job.args[2]
    t_submit = original_job.meta.get("t_submit", time.perf_counter())

    with Client.from_config() as client:
        future = client.retrieve_answer(problem_id)
        if not future.done():
            raise RuntimeError(f"Problema {problem_id} ancora in elaborazione su Leap.")
        physical_sampleset = future.sampleset

    #eseguo l'unembedding esattamente come fa FixedEmbeddingComposite
    logical_sampleset = unembed_sampleset(
        physical_sampleset, embedding, sub_bqm, chain_break_method=majority_vote
    )
    logical_sampleset.info.update(physical_sampleset.info)

    wall_service_time = time.perf_counter() - t_submit
    extra_metrics = _build_qpu_metrics(logical_sampleset, embedding, sub_bqm, wall_service_time)
    extra_metrics["recovered_from_timeout"] = True

    logical_sampleset.info["proxy_metrics"] = extra_metrics

    # Pubblica sul canale Pub/Sub atteso dal Dispatcher
    payload = pickle.dumps(logical_sampleset)
    redis_conn.publish(f"job_done:{failed_job_id}", payload)

    original_job.meta["dwave_status"] = "RECOVERED"
    original_job.save_meta()
    return {"status": "RECOVERED", "problem_id": problem_id}

def qpu_branch(sub_bqm, embedding, rq_job, chain_strength_prefactor: float = None, qpu_identity = None, real_qpu_path = False):

    if chain_strength_prefactor is not None:
                    
        chain_strength = uniform_torque_compensation(
                sub_bqm, embedding=embedding, prefactor=chain_strength_prefactor #aggiusta stima prefactor
        )
    
    else:
        chain_strength = uniform_torque_compensation(
         sub_bqm, embedding=embedding, prefactor=1.414
     )

    successLeap = False
    qpu_sampler = None
    est_us = None
    try:
        if DWAVE_API_TOKEN and real_qpu_path: 
             if qpu_identity:
                 qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity)) 
             else:
                qpu_sampler = DWaveSampler(solver={'topology__type': 'zephyr'})
                qpu_identity = qpu_sampler.solver.identity

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
                        job_label = f"rq_{rq_job.id}_att_{attempt}"

                        res_sampleset = fixed_composite.sample(
                            sub_bqm,
                            chain_strength=chain_strength,
                            num_reads=40,
                            annealing_time=20,
                            label=job_label,
                        )

                        #registro subito problem_id su Redis PRIMA dell'attesa bloccante
                        problem_id = res_sampleset.wait_id()
                        if rq_job:
                            rq_job.meta["dwave_problem_id"] = problem_id
                            rq_job.meta["dwave_status"] = "WAITING_QPU"
                            rq_job.meta["t_submit"] = t_start
                            rq_job.save_meta()

                        #attesa bloccante, interrotta in modo pulito da SIGALRM se scade il timeout RQ
                        res_sampleset.resolve()

                        t_end = time.perf_counter()
                        wall_service_time = t_end - t_start
                        successLeap = True
                        break  #successo, esco dal ciclo

                    except JobTimeoutException:
                        #eccezione attivata da segnale SIGALRM su Linux: registra lo stato e raise per attivare on_failure
                        print(f"[TIMEOUT RQ] Scaduto timeout durante attesa QPU (problem_id={rq_job.meta.get('dwave_problem_id') if rq_job else None})")
                        if rq_job:
                            rq_job.meta["dwave_status"] = "TIMED_OUT_ON_RQ"
                            rq_job.save_meta()
                        raise

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
                    extra_metrics = _build_qpu_metrics(res_sampleset, embedding, sub_bqm, wall_service_time)
                    extra_metrics['estimated_leap_qpu_time_us'] = est_us
                    actual_qat = extra_metrics['qpu_access_time']
                    charge(actual_qat)

        if not DWAVE_API_TOKEN or not successLeap:

             if DWAVE_API_TOKEN and real_qpu_path:
                 target_graph = qpu_sampler.to_networkx_graph() #uso almeno topologia reale seppur su SA
             else:
                 target_graph = dnx.zephyr_graph(12)

             t_build_0 = time.perf_counter()
             target_bqm = embed_bqm(
                 sub_bqm, 
                 embedding=embedding, 
                 target_adjacency= target_graph,
                 chain_strength=chain_strength
             )
             t_build = time.perf_counter() - t_build_0

             t_solve_0 = time.perf_counter()
             #inizializzo lo State di dwave-hybrid usando il BQM fisico (target_bqm)
             init_state = hybrid.State.from_problem(target_bqm)
             #istanzio ed eseguo il sampler da dwave-hybrid
             sa_workflow = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)
             end_state = sa_workflow.run(init_state).result()
             #estraggo il physical_sampleset risultante
             physical_sampleset = end_state.samples
             t_solve = time.perf_counter() - t_solve_0

             t_unembed_0 = time.perf_counter()
             logical_sampleset = unembed_sampleset(physical_sampleset, embedding, sub_bqm, chain_break_method=majority_vote)
             t_unembed = time.perf_counter() - t_unembed_0

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
             res_sampleset = logical_sampleset

             if "chain_break_fraction" in res_sampleset.record.dtype.names:
                 cbf = res_sampleset.record.chain_break_fraction
                 weights = res_sampleset.record.num_occurrences
                 avg_chain_break = float(np.average(cbf, weights=weights)) if len(cbf) > 0 else 0.0
             else:
                 chain_break_fraction = list(chain_break_frequency(physical_sampleset, embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.                       
                 avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

             emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, avg_chain_break)

             extra_metrics = {
                 "from_qpu": True,
                 "real_qpu": False,
                 "qpu_leap_failure": not successLeap if DWAVE_API_TOKEN and real_qpu_path else False,
                 "physical_build_time": t_build,
                 "solve_time": t_solve,
                 "unembed_time": t_unembed,
                 "estimated_leap_qpu_time_us": estimated_runtime_us,
                 "chain_break_fraction": avg_chain_break,
                 **emb_metrics
             }

    finally:
        #chiudo sempre i socket HTTP di D-Wave
        if qpu_sampler is not None:
            try:
                qpu_sampler.client.close()
            except Exception:
                pass 

    return res_sampleset, extra_metrics

def cpu_branch(sub_bqm, embedding = None):
    init_state = hybrid.State(subproblem=sub_bqm)

    t_solve_0 = time.perf_counter()
    final_state = hybrid.TabuSubproblemSampler(num_reads=20).run(init_state).result()
    t_solve = time.perf_counter() - t_solve_0

    res_sampleset = final_state.subsamples

    if embedding:
        emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, None)
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

def solve_subqubo_rq(sub_bqm: dimod.BinaryQuadraticModel,
                  sampler_type: str, embedding=None, chain_strength_prefactor: float = None, 
                  qpu_identity = None, test_type="normal", real_qpu_path = False) -> dimod.SampleSet:
        """
        Risolve direttamente il sub-QUBO con il sampler indicato.
        """
        rq_job = get_current_job()

        if test_type == "regret":
            #calcolo sia ramo QPU che CPU e calcolo regret
            route_to_qpu = (sampler_type == "qpu_queue")

            best_energy_qpu = None
            res_sampleset_qpu = None
            if embedding is not None:
                #senza embedding non posso passarlo al ramo QPU
                res_sampleset_qpu, extra_metrics_qpu = qpu_branch(sub_bqm,embedding,rq_job=rq_job,chain_strength_prefactor=chain_strength_prefactor,qpu_identity=qpu_identity, real_qpu_path=real_qpu_path)
                best_energy_qpu = res_sampleset_qpu.first.energy
            
            res_sampleset_cpu, extra_metrics_cpu = cpu_branch(sub_bqm,embedding)
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
                #non ho embedding, non posso calcolare regret energy
                extra_metrics["regret_energy_gap"] = 0.0


        elif sampler_type == "qpu_queue":
            #ho un embedding

            res_sampleset, extra_metrics = qpu_branch(sub_bqm,embedding,rq_job=rq_job,chain_strength_prefactor=chain_strength_prefactor,qpu_identity=qpu_identity, real_qpu_path=real_qpu_path)
            
        else: #cpu_queue

            res_sampleset, extra_metrics = cpu_branch(sub_bqm,embedding)

        if hasattr(res_sampleset, 'info') and isinstance(res_sampleset.info, dict):
            res_sampleset.info["proxy_metrics"] = extra_metrics

        if rq_job and rq_job.connection:
            payload = pickle.dumps(res_sampleset)
            rq_job.connection.publish(f"job_done:{rq_job.id}", payload)

        return res_sampleset


if __name__=="__main__":
    #avvio del worker RQ
    target_queue = os.getenv("TARGET_QUEUE", "cpu_queue")
    
    redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
    
    with Connection(redis_conn):
        queues_to_listen = [Queue("recovery_queue"), Queue(target_queue)] if target_queue == "qpu_queue" else [Queue(target_queue)]
        print(f"[RQ WORKER] Avvio worker in ascolto sulle code: {[q.name for q in queues_to_listen]}")
        
        #Worker effettua una fork per ogni job che viene creato
        worker = Worker(queues_to_listen)
        worker.work()