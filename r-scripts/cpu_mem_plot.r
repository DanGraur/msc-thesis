library("ggplot2")
library("dplyr")
library("reshape2")

# Parameters which are constant across all experients - START -

communication_type = "infiniband"
node_count = 4

experiment = "caffe2"
plot_folder = "plot"

# x_upper_bound = 715
by_period = 20

csv_location = switch(
  experiment,
  "caffe2" = paste('/data/files/university/msc_thesis/2nd_semester/r_images/caffe2/cpu_and_mem/', communication_type, '/n', node_count, sep = ""),
  paste('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/', communication_type, '/n', node_count, '/', experiment, sep = "")
)


has_ps = switch(
  experiment,
  "cps" = TRUE,
  "ps" = TRUE,
  "ps_n9" = TRUE,
  "ps_n17" = TRUE,
  "ps_n5" = TRUE,
  "ps_n13" = TRUE,
  "dr" = TRUE,
  "car" = FALSE,
  "dar_pscpu" = FALSE,
  "dar_ring" = FALSE,
  "caffe2" = FALSE
)

title_prefix = switch (
  experiment,
  "cps" = "Collocated Parameter Server",
  "ps" = "Parameter Server",
  "ps_n9" = "Parameter Server (9 Nodes)",
  "ps_n17" = "Parameter Server (17 Nodes)",
  "ps_n4" = "Parameter Server (4 Nodes)",
  "ps_n5" = "Parameter Server (5 Nodes)",
  "ps_n13" = "Parameter Server (13 Nodes)",
  "dr" = "Distributed Replicated",
  "car" = "Collective All Reduce",
  "dar_pscpu" = "Distributed All Reduce (PSCPU)",
  "dar_ring" = "Distributed All Reduce (RING)",
  "caffe2" = "Caffe2"
)

# Parameters which are constant across all experients - END -

# This method plots the CPU and Mem footprint given a dataset
plot_cpu_mem <- function(data, cpu_norm_factor, mem_denorm_factor, save_path, dir_name, title_prefix) {
  if (missing(cpu_norm_factor)) {
    cpu_norm_factor = 32.0 * 100.0;
  }
  
  if (missing(mem_denorm_factor)) {
    mem_denorm_factor = 64.0 / 100.0;
  }
  
  if (missing(save_path)) {
    save_path = '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_and_memory';
  } 
  
  if (missing(data)) {
    stop("data must be supplied");
  } else if (typeof(data) == 'character') {
    myData <- read.csv(data);
  } else {
    myData <- data;
  }
  
  if (missing(dir_name)) {
    dir_name = '';
  } 
  
  if (missing(title_prefix)) {
    title_prefix = '';
  } else {
    title_prefix = paste(title_prefix, " - ");
  }
  
  # Create a new column where we will record the seconds into the experiment
  myData$timestamp <- as.numeric(as.POSIXct(paste(myData$date, myData$time, sep = '-'), format="%Y-%m-%d-%H:%M:%S"));
  myData$timestamp <- myData$timestamp - min(myData$timestamp);
  
  # Normalize the CPU usage (by the number of CPUs)
  myData$true_cpu_usage <- myData$proc_perc / cpu_norm_factor;
  
  # Denormalize the memory footprint (by the total available memory)
  myData$literal_mem_usage <- myData$mem_perc * mem_denorm_factor;
  
  min_timestamp = 0;
  max_timestamp = max(myData$timestamp)
  
  # Create the CPU plot
  p <- ggplot(data = myData, aes(x = timestamp, y = true_cpu_usage)) + geom_line(color = "#00AFBB", size = 1);
  p <- p + labs(title = paste(title_prefix, 'CPU Usage'), x = "Elapsed Time [s]", y = "CPU Usage [%]");
  # p <- p + scale_x_continuous(limits = c(0, x_upper_bound), breaks = seq(0, x_upper_bound, by = by_period));
  p <- p + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  
  # Create the memory plot
  q <- ggplot(data = myData, aes(x = timestamp, y = literal_mem_usage)) + geom_line(color = "#00AFBB", size = 1);
  q <- q + labs(title = paste(title_prefix, 'Memory Footprint'), x = "Elapsed Time [s]", y = "Memory Footprint [GB]");
  # q <- q + scale_x_continuous(limits = c(0, x_upper_bound), breaks = seq(0, x_upper_bound, by = by_period));
  q <- q + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  
  # Save the plots to the specified directory
  final_path <- file.path(save_path, dir_name);
  dir.create(final_path, showWarnings = FALSE);
  
  ggsave(plot = p, filename = file.path(final_path, 'CPU_usage.png'), width = 14);
  ggsave(plot = q, filename =  file.path(final_path, 'mem_footprint.png'), width = 14);
}

# This function aggragates the mem_footprint and CPU usage into a single DF
average_datasets <- function(dir_path, join_cols, excluded_files, prefix) {
  if (missing(dir_path)) {
    dir_path = '/data/';
    warning("dir_path empty: will use '/data/' as default");
  }
  
  if (missing(join_cols)) {
    join_cols <- c("date", "time");
  }
  
  if (missing(excluded_files)) {
    excluded_files = c();
  } 
  
  if (missing(prefix)) {
    prefix = "cpu_mem_eval"
  }
  
  csv_files <- list.files(path = dir_path, pattern = paste(prefix, ".*\\.csv$" , sep = ""))
  
  final = NULL;
  
  for (file in csv_files) {
    
    if (file %in% excluded_files) {
      next;
    }
    
    current <- read.csv(file.path(dir_path, file));
  
    if (is.null(final)) {
      final <- current;
    } else {
      final <- merge(final, current, by = join_cols);
      final$proc_perc <- final$proc_perc.x + final$proc_perc.y;
      final$mem_perc <- final$mem_perc.x + final$mem_perc.y; 
      final <- select(final, c(join_cols, c("proc_perc", "mem_perc")));
    }
  }
  
  final$proc_perc <- final$proc_perc / length(csv_files);
  final$mem_perc <- final$mem_perc / length(csv_files);
 
  return(final); 
}

# This function plots the CPU usage and Memory footprint given a dir path (which contains some CSVs)
plot_multi_line_timeseries <- function(dir_path, cpu_norm_factor, mem_denorm_factor, save_path, dir_name, join_cols, has_ps, ps_name, title_prefix, prefix) {
  if (missing(cpu_norm_factor)) {
    cpu_norm_factor = 32.0 * 100.0;
  }
  
  if (missing(mem_denorm_factor)) {
    mem_denorm_factor = 64.0 / 100.0;
  }
  
  if (missing(save_path)) {
    save_path = '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_and_memory';
  } 
  
  if (missing(dir_name)) {
    dir_name = '';
  }
  
  if (missing(join_cols)) {
    join_cols <- c("date", "time");
  }
  
  if (missing(dir_path)) {
    dir_path = '/data/'
    warning("dir_path should be supplied; will default to '/data/'");
  } 
  
  if (missing(has_ps)) {
    has_ps = FALSE
    warning("No PS specified, assuming no PS is used")
  } else if (has_ps && missing(ps_name)) {
    stop("A PS CSV name must be provided")
  }
  
  if (missing(title_prefix)) {
    title_prefix = '';
  } else {
    title_prefix = paste(title_prefix, " - ");
  }
  
  if (missing(prefix)) {
    prefix = "cpu_mem_eval"
  }
  
  csv_files <- list.files(path = dir_path, pattern = paste(prefix, ".*\\.csv$", sep = ""))
  
  # This will be the graph variable
  final_df = NULL;
  
  proc_nr = 0; 
  
  for (file in csv_files) {
    current <- read.csv(file.path(dir_path, file));
    proc_nr <- proc_nr + 1;
    
    # Normalize the CPU usage (by the number of CPUs)
    current$proc_perc <- current$proc_perc / cpu_norm_factor;
    current$mem_perc <- current$mem_perc * mem_denorm_factor;
    current <- select(current, c(join_cols, c("mem_perc", "proc_perc")));
    
    # Suffix the last two column names
    colnames(current)[3] <- paste(colnames(current)[3], proc_nr, sep = "_proc_")
    colnames(current)[4] <- paste(colnames(current)[4], proc_nr, sep = "_proc_")
    
    if (is.null(final_df)) {
      final_df <- current;
    } else {
      final_df <- merge(final_df, current, by = join_cols);
    }
  }
  
  final_df$timestamp <- as.numeric(as.POSIXct(paste(final_df$date, final_df$time, sep = '-'), format="%Y-%m-%d-%H:%M:%S"));
  final_df$timestamp <- final_df$timestamp - min(final_df$timestamp);
  
  final_df_colnames <- colnames(final_df);
  cpu_df <- select(final_df, c(c("timestamp"), final_df_colnames[grepl("^proc_perc_proc", final_df_colnames)]))
  mem_df <- select(final_df, c(c("timestamp"), final_df_colnames[grepl("^mem_perc_proc", final_df_colnames)]))
  
  new_col_names <- c()
  
  if (has_ps) {
    j <- 1;
    for (i in 1:length(csv_files)) {
      if (csv_files[i] == ps_name) {
        new_col_names <- c(new_col_names, "Parameter Server");
      } else {
        new_col_names <- c(new_col_names, paste("Worker", j, sep = " "));
        j <- j + 1;
      }
    }
  } else {
    for (i in 1:length(final_df_colnames[grepl("^proc_perc_proc", final_df_colnames)])) {
      new_col_names <- c(new_col_names, paste("Worker", i, sep = " "));
    } 
  }
  
  new_col_names <- c("timestamp", new_col_names);
  colnames(cpu_df) <- new_col_names; 
  colnames(mem_df) <- new_col_names;
  
  min_timestamp = 0;
  max_timestamp = max(final_df$timestamp)
  
  cpu_data <- melt(cpu_df, id = "timestamp");
  mem_data <- melt(mem_df, id = "timestamp");
  
  final_cpu <- ggplot(cpu_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_cpu <- final_cpu + labs(title = paste(title_prefix, 'Per Process CPU Usage'), x = "Elapsed Time [s]", y = "CPU Usage [%]");
  # final_cpu <- final_cpu + scale_x_continuous(limits = c(0, x_upper_bound), breaks = seq(0, x_upper_bound, by = by_period));
  final_cpu <- final_cpu + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_cpu <- final_cpu + labs(color = "Legend"); 
  
  final_mem <- ggplot(mem_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_mem <- final_mem + labs(title = paste(title_prefix, 'Per Process Memory Footprint'), x = "Elapsed Time [s]", y = "Memory Footprint [GB]");
  # final_mem <- final_mem + scale_x_continuous(limits = c(0, x_upper_bound), breaks = seq(0, x_upper_bound, by = by_period));
  final_mem <- final_mem + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_mem <- final_mem + labs(color = "Legend");
    
  final_path <- file.path(save_path, dir_name);
  ggsave(plot = final_cpu, filename = file.path(final_path, 'proc_CPU_usage.png'), width = 14);
  ggsave(plot = final_mem, filename =  file.path(final_path, 'proc_mem_footprint.png'), width = 14);
}

plot_worker_comparison <- function(dirs, skip_files, experiment_names, true_names, cpu_norm_factor, mem_denorm_factor, save_path, file_name) {
  
  if (missing(cpu_norm_factor)) {
    cpu_norm_factor = 32.0 * 100.0;
  }
  
  if (missing(mem_denorm_factor)) {
    mem_denorm_factor = 64.0 / 100.0;
  }
  
  if (missing(save_path)) {
    save_path = '/data/files/university/msc_thesis/2nd_semester/r_images/csv';
  } 
  
  if (missing(file_name)) {
    file_name = 'comparison.csv';
  }
  
  final_dataset = NULL;
  
  for (i in 1:length(dirs)) {
    new_dataset <- average_datasets(dir_path = dirs[i], excluded_files = c(skip_files[i]));
    
    new_dataset$timestamp <- as.numeric(as.POSIXct(paste(new_dataset$date, new_dataset$time, sep = '-'), format="%Y-%m-%d-%H:%M:%S"));
    new_dataset$timestamp <- new_dataset$timestamp - min(new_dataset$timestamp);
    
    new_dataset$proc_perc <- new_dataset$proc_perc / cpu_norm_factor;
    new_dataset$mem_perc <- new_dataset$mem_perc * mem_denorm_factor;
    
    new_dataset <- select(new_dataset, c("timestamp", "proc_perc", "mem_perc"))
    
    # show(colnames(new_dataset))
    
    colnames(new_dataset)[2] <- paste(colnames(new_dataset)[2], experiment_names[i], sep = "_");
    colnames(new_dataset)[3] <- paste(colnames(new_dataset)[3], experiment_names[i], sep = "_");

    if (is.null(final_dataset)) {
      final_dataset <- new_dataset;
    } else {
      final_dataset <- merge(final_dataset, new_dataset, by = c("timestamp"), all = TRUE)
    }
  }
  
  final_col_names = c("timestamp", true_names)
  
  final_df_colnames <- colnames(final_dataset)
  cpu_df <- select(final_dataset, c(c("timestamp"), final_df_colnames[grepl("^proc_perc", final_df_colnames)]))
  mem_df <- select(final_dataset, c(c("timestamp"), final_df_colnames[grepl("^mem_perc", final_df_colnames)]))
  
  colnames(cpu_df) <- final_col_names
  colnames(mem_df) <- final_col_names
  
  cpu_data <- melt(cpu_df, id = "timestamp");
  mem_data <- melt(mem_df, id = "timestamp");
  
  
  min_timestamp = 0;
  max_timestamp = max(cpu_df$timestamp)
  
  final_cpu <- ggplot(cpu_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_cpu <- final_cpu + labs(title = "Per Process CPU Usage Comparison", x = "Elapsed Time [s]", y = "CPU Usage [%]");
  # final_cpu <- final_cpu + scale_x_continuous(limits = c(0, x_upper_bound), breaks = seq(0, x_upper_bound, by = by_period));
  final_cpu <- final_cpu + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_cpu <- final_cpu + labs(color = "Legend"); 
  
  final_mem <- ggplot(mem_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_mem <- final_mem + labs(title = "Per Process Memory Footprint Comparison", x = "Elapsed Time [s]", y = "Memory Footprint [GB]");
  # final_mem <- final_mem + scale_x_continuous(limits = c(0, x_upper_bound), breaks = seq(0, x_upper_bound, by = by_period));
  final_mem <- final_mem + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_mem <- final_mem + labs(color = "Legend");
  
  final_path <- file.path(save_path);
  ggsave(plot = final_cpu, filename = file.path(final_path, 'proc_CPU_usage.png'), width = 14);
  ggsave(plot = final_mem, filename =  file.path(final_path, 'proc_mem_footprint.png'), width = 14);
  
  # dir.create(save_path, showWarnings = FALSE);
  # write.csv(final_dataset, file.path(save_path, file_name));
}

# Location parameters - Tensorflow

# Ethernet - 12 Nodes 
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-18-17-11-02_node008_1.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-19-16-47-16_node008_0.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-19-16-47-09_node031_1.csv'),
#   "ps_n13" = c('cpu_mem_eval_2019-04-19-16-47-32_node001_0.csv'),
#   "dar_pscpu" = c('cpu_mem_eval_2019-04-18-17-11-07_node002_1.csv'),
#   "dar_ring" = c('cpu_mem_eval_2019-04-18-17-11-14_node020_0.csv'),
#   "car" = c('')
#   )

# Ethernet - 4 Nodes 
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-18-16-05-51_node055_0.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-18-16-06-32_node035_0.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-18-16-06-19_node031_1.csv'),
#   "ps_n5" = c('cpu_mem_eval_2019-04-18-16-06-47_node012_0.csv'),
#   "dar_pscpu" = c('cpu_mem_eval_2019-04-18-16-06-00_node008_1.csv'),
#   "dar_ring" = c('cpu_mem_eval_2019-04-18-16-06-11_node043_0.csv'),
#   "car" = c('')
#   )

# Infiniband - 4 Nodes 
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-18-16-46-32_10.149.0.55_1.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-18-16-47-03_10.149.0.35_0.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-18-16-46-53_10.149.0.31_0.csv'),
#   "ps_n5" = c('cpu_mem_eval_2019-04-18-16-47-33_10.149.0.12_0.csv'),
#   "dar_pscpu" = c('cpu_mem_eval_2019-04-18-16-46-40_10.149.0.8_0.csv'),
#   "dar_ring" = c('cpu_mem_eval_2019-04-18-16-46-46_10.149.0.43_1.csv'),
#   "car" = c('')
#   )

# Infiniband - 12 Nodes 
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-19-17-08-38_10.149.0.8_1.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-19-17-38-53_10.149.0.20_0.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-19-17-38-46_10.149.0.8_0.csv'),
#   "ps_n13" = c('cpu_mem_eval_2019-04-19-17-52-34_10.149.0.8_0.csv'),
#   "dar_pscpu" = c('cpu_mem_eval_2019-04-19-17-08-50_10.149.0.20_1.csv'),
#   "dar_ring" = c('cpu_mem_eval_2019-04-19-17-38-40_10.149.0.31_0.csv'),
#   "car" = c('')
#   )

### Infiniband - 8 Nodes
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/ps/',
#   "ps_n9" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/ps_n9/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n8/dar_ring/"
#   )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('0_ps.csv'),
#   "ps" = c('0_ps.csv'),
#   "ps_n9" = c(''),
#   "dr" = c('0_ps.csv'),
#   "car" = c(''),
#   "dar_pscpu" = c('cpu_mem_eval_2019-04-14-21-50-32_10.149.0.30_1.csv'),
#   "dar_ring" = c('cpu_mem_eval_2019-04-14-21-51-16_10.149.0.38_1.csv')
#   )

### Infiniband - 16 Nodes
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/ps/',
#   "ps_n9" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/ps_n17/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/csv/infiniband/n16/dar_ring/"
#   )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-14-23-05-40_10.149.0.30_0.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-14-23-05-20_10.149.0.8_0.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-14-23-04-53_10.149.0.46_0.csv'),
#   "dar_pscpu" = c('cpu_mem_eval_2019-04-14-22-06-08_10.149.0.8_0.csv'),
#   "dar_ring" = c('cpu_mem_eval_2019-04-14-22-06-21_10.149.0.46_0.csv'),
#   "ps_n9" = c('cpu_mem_eval_2019-04-14-23-43-06_10.149.0.8_0.csv'),
#   "car" = c('')
#   )

### Ethernet - 8 Nodes
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/ps/',
#   "ps_n9" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/ps_n9/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/dar_ring/"
#   )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-10-12-59-10_node053_0.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-10-13-04-57_node030_0.csv'),
#   "ps_n9" = c('cpu_mem_eval_2019-04-11-13-09-54_node053_0.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-10-14-15-45_node053_0.csv'),
#   "car" = c(),
#   "dar_pscpu" = c(),
#   "dar_ring" = c()
#   )

### Ethernet - 16 Nodes
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/n16/cps/',
#   "ps_n17" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/n16/ps_n17/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/n16/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/n16/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/n16/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/csv/n16/dar_ring/"
# )
#
# skip_files = switch(
#   experiment,
#   "cps" = c('cpu_mem_eval_2019-04-10-16-39-30_node008_0.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-10-16-39-11_node030_0.csv'),
#   "ps_n17" = c('0_ps.csv'),
#   "dr" = c('cpu_mem_eval_2019-04-10-16-24-13_node008_1.csv'),
#   "car" = c(),
#   "dar_pscpu" = c(),
#   "dar_ring" = c()
#   )

# Skip files for n8 redoux
# skip_files = switch (experiment,
#   "cps" = c('cpu_mem_eval_2019-04-11-17-27-53_node053_0.csv'),
#   "ps" = c('cpu_mem_eval_2019-04-11-17-27-26_node030_0.csv'),
#   "car" = c()
# )


# dirs = c(
#   '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/ps',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/ps_n9',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/cps',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/car'
# )
# 
# excluded_files = c(
#   "cpu_mem_eval_2019-04-10-13-04-57_node030_0.csv",
#   "cpu_mem_eval_2019-04-11-13-09-54_node053_0.csv",
#   "cpu_mem_eval_2019-04-10-12-59-10_node053_0.csv",
#   ""
# )
# 
# experiment_names = c(
#   "ps",
#   "ps_n9",
#   "cps",
#   "car"
# )
# 
# true_names = c(
#   "Parameter Server",
#   "Parameter Server (9 Nodes)",
#   "Collocated Parameter Server",
#   "Collective All Reduce"
# )
# 
# plot_worker_comparison(dirs = dirs, skip_files = excluded_files, experiment_names = experiment_names, true_names = true_names)
# 
# ps <- average_datasets(dir_path = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/ps', excluded_files = c('cpu_mem_eval_2019-04-10-13-04-57_node030_0.csv'))
# ps_n9 <- average_datasets(dir_path = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/ps_n9', excluded_files = c('cpu_mem_eval_2019-04-11-13-09-54_node053_0.csv'))
# cps <- average_datasets(dir_path = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/cps', excluded_files = c('cpu_mem_eval_2019-04-10-12-59-10_node053_0.csv'))
# car <- average_datasets(dir_path = '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/car/', excluded_files = c())
# 
# write.csv(ps, '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/normalized_x_plots/comparison/ps.csv')
# write.csv(cps, '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/normalized_x_plots/comparison/cps.csv')
# write.csv(ps_n9, '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/normalized_x_plots/comparison/ps_n9.csv')
# write.csv(car, '/data/files/university/msc_thesis/2nd_semester/r_images/csv/ethernet/n8/normalized_x_plots/comparison/car.csv')

# Location parameters - Caffe2

skip_files = c('');

# Plot the aggregate timeseries
plot_cpu_mem(data = average_datasets(dir_path = csv_location, excluded_files = skip_files), dir_name = plot_folder, save_path = csv_location, title_prefix = title_prefix);
# Plot the individual timeseries
plot_multi_line_timeseries(dir_path = csv_location, save_path = csv_location, dir_name = plot_folder, has_ps = has_ps, ps_name = skip_files[1], title_prefix = title_prefix);
