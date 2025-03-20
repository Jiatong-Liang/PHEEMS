def main():
    import jax.experimental.sparse as jesp
    import networkx as nx
    import scipy.sparse as sp
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import msprime as msp
    import networkx as nx
    from parameters_EEMS_anc import MCMCParams
    import random
    from phlash.data import TreeSequenceContig

    def create_demography(Ne, m, sample_size, q_anc, tau, max_q = 2, max_m = 0.1):
        test_m = []
        test_q = []
        testing_pairs = []
        pairs = []
        edges = []
        demo = msp.Demography()
        G = nx.Graph()
        num_rows = 2
        num_cols = 4
        # q = 2
        multiplier = sample_size * 2
        demo.add_population(initial_size= (2*Ne) / q_anc, name = "anc")
        
        for i in range(num_rows * num_cols):
            pop_name = f"P{i}"
            q = random.uniform(0, max_q)
            test_q.append(q)
            demo.add_population(initial_size= (2*Ne) / q, name=pop_name)
            testing_pairs.append((i, i))

            while True:
                pair = (random.randint(i * multiplier, (i + 1) * multiplier - 1),
                        random.randint(i * multiplier, (i + 1) * multiplier - 1))
                # Ensure the two numbers are not the same
                if pair[0] != pair[1]:
                    pairs.append(pair)
                    break  # Exit while loop once a valid pair is found

        tmp = [f"P{i}" for i in range(num_rows*num_cols)]
        demo.add_population_split(time = tau * 4 * Ne, derived=tmp, ancestral="anc")
        
        # Create 200 populations in a 20x10 grid with random scaling factors for q
        for i in range(num_rows):
            for j in range(num_cols):
                pop_index = i * num_cols + j  # Unique index for each population
                pop_name = f"P{pop_index}"
                    # Prefix with "P" to make valid Python identifiers

                # Generate a random scaling factor q between 1 and 5
                
                # Add population to demography with scaled size based on random q
                G.add_node(pop_index)

                # Set migration rates to adjacent populations (right and down neighbors)
                # Right neighbor
                if j < num_cols - 1:
                    right_neighbor = f"P{pop_index + 1}"
                    # m = random.uniform(0, max_m)
                    m = 0.01
                    demo.set_symmetric_migration_rate(
                        populations=(pop_name, right_neighbor), rate=m / (4 * Ne))
                    test_m.append(m)
                    G.add_edge(pop_index, pop_index + 1)
                    edges.append((pop_index, pop_index + 1))
                    testing_pairs.append((pop_index, pop_index+1))
                    pairs.append((random.randint(pop_index*multiplier, (pop_index+1)*multiplier - 1), 
                                random.randint((pop_index+1)*multiplier, (pop_index+2)*multiplier - 1)))


                # Bottom neighbor
                if i < num_rows - 1:
                    bottom_neighbor = f"P{pop_index + num_cols}"
                    # m = random.uniform(0, max_m)
                    m = 0.1
                    demo.set_symmetric_migration_rate(
                        populations=(pop_name, bottom_neighbor), rate=m / (4 * Ne))
                    test_m.append(m)
                    G.add_edge(pop_index, pop_index + num_cols)
                    edges.append((pop_index, pop_index + num_cols))
                    testing_pairs.append((pop_index, pop_index+num_cols))
                    pairs.append((random.randint(pop_index*multiplier, (pop_index+1)*multiplier - 1), 
                                    random.randint((pop_index+num_cols)*multiplier, (pop_index+num_cols+1)*multiplier - 1)))


        # Set up sample sizes for each population using prefixed identifiers
        samples = {f"P{i}": sample_size for i in range(num_rows * num_cols)}
        # print(samples)
        anc = msp.sim_ancestry(samples=samples, demography=demo, recombination_rate=1e-8, sequence_length=1e7)
        ts = msp.sim_mutations(anc, rate=1e-8)
        return G, demo, test_q, test_m, testing_pairs, pairs, num_cols*num_rows, ts, edges

    # Example usage
    Ne = 1e4           # Effective population size
    m = 0.1            # Fixed migration rate between all adjacent populations
    sample_size = 1  # Sample size for each population

    # Generate the demography and graph for the 20x10 grid
    q_anc = 1
    tau = 3.0
    basic_graph, demo, test_q, test_m, testing_pairs, pairs, total_nodes, ts, edges = create_demography(Ne, m, sample_size, q_anc, tau)

    for i in range(30):
        print(ts.samples([i]))
    # [1.353412681173156, 1.2763824076566788, 1.2447130796641515, 0.11414226841290365]
    print(test_q)
    print(test_m)

    # Testing pairs is the DEMES pairing like (0,0) means both samples came from population 0
    print(testing_pairs)
    # pairs is the haploid ID that we sample, these cannot have duplicate values
    print(pairs)
    # Note that both pairs and testing_pairs eliminated all possibility of having 
    # samples from DIFFERENT nodes because JT's old_phlash doesn't allow for duplicate samples IDs
    tmp = jesp.BCOO.from_scipy_sparse(sp.triu(nx.adjacency_matrix(basic_graph, np.arange(total_nodes)).astype(float)))
    print(tmp.indices)
    # Recall that we need to construct a TreeSequence object OUTSIDE of the MCMC file now because we construct
    # the mapping before we call fit function
    TreeSeq = TreeSequenceContig(ts, nodes = pairs)

    mutation_rate = 1e-8
    Ne = 1e4
    t1 = 1e-4
    tM = 15.0
    window_size = 100
    theta = 4 * Ne * mutation_rate
    rho = 1.0 * theta
    pat = "14*1+1*2"

    # obtaining upper triangular of adjacency matrix, cannot use np.triu because we are
    # working with a sparse matrix, not a numpy matrix
    initialization = MCMCParams.from_linear(
            pattern=pat,
            rho=rho * window_size,
            t1 = t1, 
            tM = tM,
            BCOO_indices = tmp.indices,
            m = np.ones(len(test_m)), 
            q = np.ones(len(test_q)),
            tau=1.0,
            q_anc = 1.0,
            theta=theta * window_size,
            alpha=0.0,
            beta=0.0,
        )

    from MCMC_Method_EEMS_anc import running_MCMC
    from utility_methods_old import chunk_attributes, chunk_map
    num_node_pairs, num_chunks, chunk_length = chunk_attributes(TreeSeq, window_size=100, overlap=500)
    map = chunk_map(testing_pairs, num_chunks)
    print(map)

    particles = running_MCMC(initialization, TreeSeq, testing_pairs, map, num_particles=10)
    print(test_q)
    
if __name__ == "__main__":
    main()
[1.7697383983881945, 1.432974147540729, 0.42596829582058127, 0.4604975839571501, 0.7977356386216805, 0.5237559478629683, 0.7984529562348608, 1.4537882307539287]
[0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.1, 0.01, 0.01, 0.01]