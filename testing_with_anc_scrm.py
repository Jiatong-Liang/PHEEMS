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

    def create_demography(Ne, m, sample_size, q_anc, tau):
        test_m = []
        test_q = []
        testing_pairs = []
        pairs = []
        demo = msp.Demography()
        G = nx.Graph()
        num_rows = 4
        num_cols = 4
        # q = 2
        multiplier = sample_size * 2
        demo.add_population(initial_size=2*Ne / q_anc, name = "anc")
        
        for i in range(num_rows * num_cols):
            pop_name = f"P{i}"
            q = random.uniform(0, 2)
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
                    demo.set_symmetric_migration_rate(
                        populations=(pop_name, right_neighbor), rate=m / (4 * Ne))
                    test_m.append(m)
                    G.add_edge(pop_index, pop_index + 1)
                    # testing_pairs.append((pop_index, pop_index+1))
                    pairs.append((random.randint(pop_index*multiplier, (pop_index+1)*multiplier - 1), 
                                random.randint((pop_index+1)*multiplier, (pop_index+2)*multiplier - 1)))


                # Bottom neighbor
                if i < num_rows - 1:
                    bottom_neighbor = f"P{pop_index + num_cols}"
                    demo.set_symmetric_migration_rate(
                        populations=(pop_name, bottom_neighbor), rate=m / (4 * Ne))
                    test_m.append(m)
                    G.add_edge(pop_index, pop_index + num_cols)
                    # testing_pairs.append((pop_index, pop_index+num_cols))
                    pairs.append((random.randint(pop_index*multiplier, (pop_index+1)*multiplier - 1), 
                                    random.randint((pop_index+num_cols)*multiplier, (pop_index+num_cols+1)*multiplier - 1)))


        # Set up sample sizes for each population using prefixed identifiers
        samples = {f"P{i}": sample_size for i in range(num_rows * num_cols)}
        # print(samples)
        return G, demo, test_q, test_m, testing_pairs, pairs, num_cols*num_rows

    # Example usage
    Ne = 1e4           # Effective population size
    m = 0.05            # Fixed migration rate between all adjacent populations
    sample_size = 1   # Sample size for each population

    # Generate the demography and graph for the 20x10 grid
    q_anc = 2
    tau = 0.5
    basic_graph, demo, test_q, test_m, testing_pairs, pairs, total_nodes = create_demography(Ne, m, sample_size, q_anc, tau)
    print(demo)

    import demes
    cmd = demes.ms.to_ms(demo.to_demes(), N0=1e4, samples=[2] * (total_nodes+1))

    L = 1e7
    theta = 4 * 1e-8 * Ne * L
    r = 4 * 1e-8 * Ne * L

    # you use os if you don't care to store the output from the terminal
    # all I need is the output.txt which I can create and read anyways so I use os
    import os
    command = f"/Users/jkliang/Desktop/EEMS_Old/scrm-1.7.4/scrm {2*(total_nodes+1)} 1 -t {theta} -r {r} {L} --transpose-segsites -SC abs -p 14 -oSFS {cmd} > /Users/jkliang/Desktop/EEMS_Old/output.txt"
    os.system(command)
    from phlash.sim import _parse_scrm
    from phlash.data import VcfContig
    with open("/Users/jkliang/Desktop/EEMS_Old/output.txt", "r") as scrm_out:
        vcf = _parse_scrm(scrm_out, chrom_name="chr1")

    open("/Users/jkliang/Desktop/EEMS_Old/output.vcf", "wt").write(vcf)
    command = f"bcftools view output.vcf -o /Users/jkliang/Desktop/EEMS_Old/output.bcf"
    os.system(command)
    command = f"bcftools index /Users/jkliang/Desktop/EEMS_Old/output.bcf"
    os.system(command)
    samples = [f"sample{i}" for i in range(total_nodes)]
    TreeSeq = VcfContig("/Users/jkliang/Desktop/EEMS_Old/output.bcf", samples=samples, contig = "chr1", interval = (0, 1e7))
###
    print(test_q)

    print(testing_pairs)
    tmp = jesp.BCOO.from_scipy_sparse(sp.triu(nx.adjacency_matrix(basic_graph, np.arange(total_nodes)).astype(float)))
    print(tmp.indices)


    print(TreeSeq.get_data(100))
    print(TreeSeq.L)
    # [10865.  7322. 34116.  3470.  2342.]

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
