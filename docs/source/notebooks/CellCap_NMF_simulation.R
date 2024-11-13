library(fastTopics)

pert_simulation_wNMF <- function(n_cells=2000, n_genes=1000, n_factors=30,
                                 n_pert_factors=2, n_basal_factors=10,
                                 n_pert=30, alpha=5.0, 
                                 partition=5, if_partition=FALSE) {
  
  nmf_sim <- simulate_count_data(n=n_cells, m=n_genes, k=n_factors)
  
  Fbasal <- nmf_sim$F[,1:n_basal_factors]
  Lbasal <- nmf_sim$L[,1:n_basal_factors]
  
  Fpert <- nmf_sim$F[,(n_basal_factors+1):n_factors]
  Lpert <- nmf_sim$L[,(n_basal_factors+1):n_factors]
  
  pert_index <- matrix(c(1:(n_factors - n_basal_factors)), nrow = partition, 
                       byrow = TRUE)
  sample_index <- matrix(c(1:n_pert), nrow = partition, byrow = TRUE)
  
  simulated <- basal_ct <- Lbasal %*% t(Fbasal)
  
  gt_response <- sample(c(1:n_basal_factors), n_pert, replace = TRUE)
  gt_pert <- list()
  for (i in c(1:n_pert_factors)) {
    if (if_partition){
      sample_pert <- c()
      for (j in c(1:n_pert)) {
        index <- which(sample_index == j, arr.ind=TRUE)[1]
        sample_pert <- c(sample_pert, sample(pert_index[index,], 1, 
                                             replace = TRUE))
      }
      gt_pert[[i]] <- sample_pert
    }
    else{
      gt_pert[[i]] <- sample(c(1:(n_factors - n_basal_factors)), 
                             n_pert, replace = TRUE)
    }
  }
  
  design_matrix <- gt_response
  for (i in c(1:n_pert_factors)) {
    design_matrix <- cbind(design_matrix, as.integer(gt_pert[[i]]))
  }
  design_matrix <- data.frame(design_matrix)
  
  basal_response <- Lbasal[,gt_response[1]]
  basal_response[basal_response != 0 ] <- 0
  
  ## make sure control cells have the same sum of counts
  pert_count <- matrix(rep(1, times = nrow(Lbasal) * n_pert_factors), 
                       nrow = nrow(Lbasal), ncol = n_pert_factors)
  pert_count <- pert_count %*% t(Fbasal[,1:n_pert_factors])
  simulated <- simulated + pert_count * alpha # * Lbasal[,gt_response[1]] 
  
  for (i in c(1:n_pert)) {
    pert_list <- c()
    for (j in c(1:n_pert_factors)) {
      pert_list <- c(pert_list, gt_pert[[j]][i])
    }
    
    # pert_count <- Lpert[,pert_list] %*% t(Fpert[,pert_list])
    pert_count <- matrix(rep(1, times = nrow(Lpert) * n_pert_factors), 
                         nrow = nrow(Lpert), ncol = n_pert_factors)
    pert_count <- pert_count %*% t(Fpert[,pert_list])
    
    pert_count <- basal_ct + pert_count * Lbasal[,gt_response[i]] * alpha
    simulated <- rbind(simulated, pert_count)
    basal_response <- rbind(basal_response, Lbasal[,gt_response[i]])
  }
  
  # poisson_sample_matrix <- as.data.frame(lapply(simulated, function(col) {
  #   rpois(length(col), col)
  # }))
  
  return(list('data'=simulated, 'dm'=design_matrix, 
              'basal'=basal_response, 'perturb'=Fpert))
  
}

##Run to generate simulation data
nmf_sim <- pert_simulation_wNMF(n_pert=20, alpha=2.0, n_pert_factors = 2)

write.csv(nmf_sim$data,
          '../NMF_simulated_data.csv')
write.csv(nmf_sim$dm,
          '../NMF_simulated_design_matrix.csv')
write.csv(nmf_sim$basal,
          '../NMF_simulated_basal_response.csv')
write.csv(nmf_sim$perturb,
          '../NMF_simulated_perturb_factor.csv')
