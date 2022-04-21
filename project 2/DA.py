import sklearn as skl
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidian",
                T_min = 0.1, T0=1e-2, alpha=0.95, tol=1e-4, verbose = True):
                
        self.n_clusters = n_clusters     # maximal number of clusters
        self.random_state = random_state
        self.metric = metric
        self.T_min = T_min 
        self.T0 = T0
        self.alpha = alpha
        self.tol = tol
        self.verbose = verbose        
        self.T = None
        self.cluster_centers = None
        self.cluster_probs = None
        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()

    # critical temperature calculation for cluster k
    def critical_temp(self, samples, k, p_x):
        p_y =  self.p_y_ls[k]
        delta = samples - self.cluster_centers[k]
        n_features = samples.shape[1]
        C_xy = np.zeros((n_features,n_features))
        
        for x in range(samples.shape[0]):
            # Bayes Theorem  p(x_i|y) = p(x_i)* p(y|x_i)/ p(y)
            p_x_y = np.dot(p_x, self.cluster_probs[x, k])/ p_y
            diff = delta[x,:].reshape(1,-1)
            C_xy += p_x_y * np.matmul(diff.T, diff)
            
        max_ev = np.max(np.linalg.eigvals(C_xy))
        T_crit = 2* max_ev
        
        return T_crit   
        
    
    # determining the free energy and distortion
    def distortion_free_E(self, samples, n_clusters):
        T = self.T_min
        n_samples = samples.shape[0]
        p_x = 1.0 / n_samples
        distance_sq = np.array([np.linalg.norm(samples - self.cluster_centers[k], axis=1)**2 for k in range(n_clusters)]).T
        p_joint = self.cluster_probs * p_x
        distortion = np.sum(p_joint *distance_sq)
        H = - np.sum(p_joint * np.log(p_joint + 1e-8))
        E = distortion - T * H
        
        return distortion, E
 

    def split_bifurcation_tree(self, k):
        k_current = len(self.bifurcation_tree.leaves())
        parent_id = [x.identifier for x in self.bifurcation_tree.leaves() if x.tag == k][0]
        l = str(int(parent_id.split('_')[-1]) + 1)
        l1 = str(k) + '_' + l
        l2 = str(k_current) + '_' + l
        self.bifurcation_tree.create_node(k, l1, parent=parent_id)
        self.bifurcation_tree.create_node(k_current, l2, parent=parent_id)
    

    def fit(self, samples):
        """Compute DAC for input vectors samples

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """

        K_max = self.n_clusters  # set maximal number of cluster
        alpha = self.alpha           # cooling factor 
                
        if self.metric == "euclidian":
            
            self.split_temp = []
            n_samples = samples.shape[0]
            n_features = samples.shape[1]
            p_x = 1.0 / n_samples
            
            # initializiation
            K_curr = 1                      # current number of clusters
            self.n_eff_clusters.append(K_curr)
            y = np.mean(samples, axis=0)
            
            self.cluster_centers = y.reshape(1,-1) 
            # initialize cluster assignment probability to uniform
            self.cluster_probs = np.ones((n_samples, 1), dtype='float32')
            self.p_y_ls = np.ones((1, 1), dtype='float32') 
            
            # initialize temperature slightly above critical one
            T = self.critical_temp(samples, 0, p_x)*1.01
            
            distortion, energy = self.distortion_free_E(samples, K_curr)
            self.temperatures.append(T)
            self.distortions.append(distortion)
            
            # Initiliaze bifurcation tree (SECTION 5.0)
            self.bifurcation_tree = Tree()
            self.bifurcation_tree.create_node(0, '0_0')
            self.tree_refs = [self.cluster_centers[0]]
            self.tree_dirs = [1]
            self.tree_dists = [[] for x in range(self.n_clusters)]
            self.tree_temperatures = [[] for x in range(self.n_clusters)]
            self.tree_dists[0].append(self.tree_dirs[0] * (np.linalg.norm(self.tree_refs[0] - self.cluster_centers[0])))
            self.tree_temperatures[0].append(T)
            self.tree_offsets = [0.0]
            
            t_loop_counter = -1
            
            # loop over temperatures 
            while T >= self.T_min:
                
                t_loop_counter += 1
                if self.verbose: print('counter: {} -- temperature : {}'.format(t_loop_counter,T))               
                
                # Loop for EM-Algorithm
                EM_counter = -1
                while True:
                    EM_counter += 1
                    
                    # STEP 3
                    # E-step
                    prev_centroids = self.cluster_centers.copy()  # previous centroids
                    distance_sq = self.get_distance(samples, prev_centroids)
                    self.cluster_probs = self._calculate_cluster_probs(distance_sq, T)

                    # M-step
                    p_y_x = self.cluster_probs 
                    self.p_y_ls = []

                    for k in range(K_curr):
                        p_y_x_k = p_y_x[:,k]
                        p_y = p_x*np.sum(p_y_x_k)
                        y = np.sum(p_x * np.expand_dims(p_y_x_k , axis=1) * samples , axis=0)/p_y   # (6000,) (6000,2) 
                        self.p_y_ls.append(p_y)
                        self.cluster_centers[k] = y
                    self.p_y_ls = np.expand_dims(self.p_y_ls, axis=1)

                    # convergence test - (STEP 4)
                    distortion, energy = self.distortion_free_E(samples, K_curr)

                    error = np.linalg.norm(self.cluster_centers - prev_centroids)
                    # break if EM-algorithm reached convergence criterion 
                    if  error < self.tol: break 
                
                    if self.verbose: print('-'*10 + 'K_curr {} -- distortion {} -- temperature {} -- error {} '.format(K_curr,distortion,T, error) +'-'*10)

                # Stop algorithm (loop over temperatures) if minimal temperature is reached
                if T == self.T0:
                    print('Minimal temperature reached.')
                    break
                    
                # check temperature threshold (STEP 5)
                if T <= self.T_min:
                    print('Threshold reached - last iteration will be performed.')
                    T = self.T0  # set temperature to minimal one
                    continue
                    
                # perform cooling (STEP 6)
                T *= self.alpha 

                # Section 4.5 - update lists needed for phase diagram                
                distortion, energy = self.distortion_free_E(samples, K_curr)
                self.temperatures.append(T)
                self.distortions.append(distortion)
              
                ###################################################
                
                # check for phase transition (STEP 7)           
                if K_curr < self.n_clusters :
                    # Check critical temperatures
                    for k in range(0, K_curr):
                        
                        # critical temperature
                        T_crit = self.critical_temp(samples, k, p_x)  
                        if self.verbose: print('Temp: {}, Crit Temp: {}, Number of Cl: {}, current Cl {}: '.format(T, T_crit, K_curr, k))
                        
                        if T < T_crit and K_curr < self.n_clusters:
                            
                            self.split_temp.append(T)
                            prev = self.cluster_centers[k]
                            np.random.seed(seed=self.random_state)
                            noise = np.random.normal(0, 0.1, size = n_features)
                            new_centroid = self.cluster_centers[k] + noise
                            self.cluster_centers = np.vstack((self.cluster_centers, new_centroid))
                            self.p_y_ls = self.p_y_ls[:,0]
                            self.p_y_ls = np.append(self.p_y_ls, [self.p_y_ls[k]/2])
                            self.p_y_ls[k] = self.p_y_ls[k]/2
                            self.p_y_ls = np.expand_dims(self.p_y_ls, axis=1)
                            
                            if self.verbose: 
                                print("*"*100)
                                print('Centroid {} of {} is split at temperature {} with critical temperature {}'.format(k, K_curr, T, T_crit))
                                print("*"*100)
                            K_curr += 1
                            
                            self.split_bifurcation_tree(k)
                            self.tree_dirs[k] = -1
                            self.tree_dirs.append(1)
                            self.tree_refs[k] = self.cluster_centers[k].copy()
                            self.tree_refs.append(self.cluster_centers[k].copy())
                            dist = np.linalg.norm(self.tree_refs[k] - self.cluster_centers[k])
                            self.tree_offsets[k] = self.tree_dists[k][-1]
                            self.tree_offsets.append(self.tree_dists[k][-1])
                            self.tree_dists[k].append(self.tree_offsets[k] + self.tree_dirs[k] * dist)
                            self.tree_temperatures[k].append(T)
                            self.tree_dists[K_curr-1].append(self.tree_offsets[k] + self.tree_dirs[k] * dist)
                            self.tree_temperatures[K_curr-1].append(T)
                        
                        else:
                            dist = np.linalg.norm(self.tree_refs[k] - self.cluster_centers[k])
                            self.tree_dists[k].append(self.tree_offsets[k] + self.tree_dirs[k] * dist)
                            self.tree_temperatures[k].append(T)

                self.n_eff_clusters.append(K_curr)

        return self                                  
        
    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        # We assume that dist_mat contains squared! distances
        n_samples = dist_mat.shape[0]
        n_clusters = dist_mat.shape[1]
        
        curr_probs = self.cluster_probs          
        p_y = self.p_y_ls 
        
        probs = np.zeros((n_samples, n_clusters ))
        Z = np.zeros((n_samples, ))  # normalizer
        min_dist = np.min(dist_mat, axis=1, keepdims=True)
        weights = np.multiply(np.exp((-dist_mat + min_dist)/temperature) , self.p_y_ls.T)
        probs = weights / np.sum(weights, axis=1, keepdims=True)
            
        return probs
        

    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        
        metric = self.metric
        n_clusters = clusters.shape[0]
        distance_sq = np.zeros((samples.shape[0], n_clusters))

        if metric == "euclidian":
            distance_sq = np.array([np.linalg.norm(samples - clusters[k], axis=1)**2 for k in range(n_clusters)]).T

        return distance_sq

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        """
        check_is_fitted(self, ["bifurcation_tree"])
        
        plt.figure(figsize=(10, 5))
        for k in range(self.n_clusters):
            plt.plot(self.tree_dists[k], self.tree_temperatures[k], label=str(k))
            
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()
    

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, y_axis)
        
        region = {}

        for i, c in list(enumerate(self.n_eff_clusters)):

            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        
        for c in region:
            if c == 0:
                continue
            if c < np.max(list(region.keys())): region[c]['max'] = region[c+1]['min']
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()