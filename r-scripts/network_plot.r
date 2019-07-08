library("reshape2")
library("ggplot2")
library("dplyr")

### START: These globals are constant regardless of experiment

communication_type = "infiniband"
node_count = 12

experiment = "caffe2"
plot_directory = 'plot'


csv_location = switch(
  experiment,
  "caffe2" = paste('/data/files/university/msc_thesis/2nd_semester/r_images/caffe2/network/', communication_type, '/n', node_count, sep = ""),
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
  # DAR architectures use a controller
  "dar_pscpu" = TRUE,
  "dar_ring" = TRUE,
  "caffe2" = FALSE
)

collocated_ps = switch (
  experiment,
  "cps" = TRUE,
  "dr" = TRUE,
  FALSE
)

title_prefix = switch (
  experiment,
  "cps" = "Collocated Parameter Server",
  "ps" = "Parameter Server",
  "ps_n9" = "Parameter Server (9 Nodes)",
  "ps_n17" = "Parameter Server (17 Nodes)",
  "ps_n4" = "Parameter Server (4 Nodes)",
  "ps_n16" = "Parameter Server (16 Nodes)",
  "ps_n5" = "Parameter Server (5 Nodes)",
  "ps_n13" = "Parameter Server (13 Nodes)",
  "dr" = "Distributed Replicated",
  "car" = "Collective All Reduce",
  "dar_pscpu" = "Distributed All Reduce (PSCPU)",
  "dar_ring" = "Distributed All Reduce (RING)",
  "caffe2" = "Caffe2"
)

by_period = 40


### END: These globals are constant regardless of experiment


# This function aggragates the tx_kbps and rx_kbps into a single DF
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
    prefix = "network_traffic"
  }
  
  csv_files <- list.files(path = dir_path, pattern = paste(prefix, ".*\\.csv$", sep = ""));
  
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
      final$tx_kbps <- final$tx_kbps.x + final$tx_kbps.y;
      final$rx_kbps <- final$rx_kbps.x + final$rx_kbps.y; 
      final <- select(final, c(join_cols, c("tx_kbps", "rx_kbps")));
    }
  }
  
  final$tx_kbps <- final$tx_kbps / length(csv_files);
  final$rx_kbps <- final$rx_kbps / length(csv_files);
  
  return(final); 
}

# Add cumulative columns
add_cumulative_columns <- function(current, col_names) {
  if (missing(current)) {
    stop("No dataset passed");
  }
  
  if (missing(col_names)) {
    col_names = c("tx_kbps", "rx_kbps");
    warning("No column names passed");
  }
  
  for (col_idx in 1:length(col_names)) {
    current_col <- paste(col_names[col_idx], "accumulated", sep = "_");
      
    current[current_col] = 0;
    current[1, current_col] = current[1, col_names[col_idx]];
      
    for (row_idx in 2:nrow(current)) {
      current[row_idx, current_col] <- current[row_idx - 1, current_col] + current[row_idx, col_names[col_idx]];
    }
  }
  
  return(current);
}

# This method plots the CPU and Mem footprint given a dataset
plot_tx_rx <- function(data, data_multiplier, col_names, save_path, save_dir_name, title_prefix) {
  if (missing(data)) {
    stop("data must be supplied");
  } else if (typeof(data) == 'character') {
    myData <- read.csv(data);
  } else {
    myData <- data;
  }
  
  if (missing(data_multiplier)) {
    #  1.0 means kbyte, 1024.0 is byte, etc. 
    data_multiplier = 1.0; 
  }
  
  if (missing(save_path)) {
    save_path = '/data/files/university/msc_thesis/2nd_semester/r_images/network';
  } 
  
  if (missing(save_dir_name)) {
    save_dir_name = '';
  } 
  
  if (missing(title_prefix)) {
    title_prefix = '';
  } else {
    title_prefix = paste(title_prefix, " - ");
  }
  
  if (missing(col_names)) {
    col_names = c("tx_kbps", "rx_kbps");
  }
  
  # Find the extended list of column names (i.e. those which contain accumulated)
  extended_col_names = c(col_names);
  
  for (i in 1:length(col_names)) {
    extended_col_names = c(extended_col_names, paste(col_names[i], "accumulated", sep = "_"))
  }
  
  # Normalize the traffic columns
  for (col_name in extended_col_names) {
    myData[col_name] = myData[col_name] * data_multiplier;
  }
  
  # Create a new column where we will record the seconds into the experiment
  myData$timestamp <- as.numeric(as.POSIXct(paste(myData$date, myData$time, sep = '-'), format="%Y-%m-%d-%H:%M:%S"));
  myData$timestamp <- myData$timestamp - min(myData$timestamp);
  
  final_df_colnames <- colnames(myData);
  tx_df <- select(myData, c(c("timestamp"), final_df_colnames[grepl("^tx_kbps", final_df_colnames)]))
  rx_df <- select(myData, c(c("timestamp"), final_df_colnames[grepl("^rx_kbps", final_df_colnames)]))
  
  colnames(tx_df) <- c("timestamp", "Tx kilobits", "Accumulated Tx kilobits")
  colnames(rx_df) <- c("timestamp", "Rx kilobits", "Accumulated Rx kilobits")
  
  tx_data <- melt(tx_df, id = "timestamp");
  rx_data <- melt(rx_df, id = "timestamp");
  
  min_timestamp = 0;
  max_timestamp = max(tx_data$timestamp)
  
  # Create the TX
  final_tx <- ggplot(tx_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_tx <- final_tx + labs(title = paste(title_prefix, 'Sent Bits'), x = "Elapsed Time [s]", y = "Kilobits");
  final_tx <- final_tx + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_tx <- final_tx + labs(color = "Legend"); 
  
  # Create the memory plot
  final_rx <- ggplot(rx_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_rx <- final_rx + labs(title = paste(title_prefix, 'Received Bits'), x = "Elapsed Time [s]", y = "Kilobits");
  final_rx <- final_rx + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_rx <- final_rx + labs(color = "Legend"); 
  
  
  # Save the plots to the specified directory
  final_path <- file.path(save_path, save_dir_name);
  dir.create(final_path, showWarnings = FALSE);
  
  ggsave(plot = final_tx, filename = file.path(final_path, 'TX_Bits.png'), width = 14);
  ggsave(plot = final_rx, filename =  file.path(final_path, 'RX_Bits.png'), width = 14);
}

# This function plots the CPU usage and Memory footprint given a dir path (which contains some CSVs)
plot_multi_line_timeseries_accumulated <- function(dir_path, data_multiplier, save_path, dir_name, join_cols, has_ps, ps_name, collocated_ps, title_prefix, prefix) {
  if (missing(data_multiplier)) {
    #  1.0 means kbyte, 1024.0 is byte, etc. 
    data_multiplier = 1.0; 
  }
  
  if (missing(save_path)) {
    save_path = '/data/files/university/msc_thesis/2nd_semester/r_images/network';
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
    has_ps = FALSE;
    warning("No PS specified, assuming no PS is used");
  } else if (has_ps && missing(ps_name)) {
    stop("Must specify the name of the PS CSV");
  }
  
  if (missing(collocated_ps)) {
    collocated_ps <- FALSE
  }
  
  if (missing(title_prefix)) {
    title_prefix = '';
  } else {
    title_prefix = paste(title_prefix, " - ");
  }
  
  if (missing(prefix)) {
    prefix = "network_traffic"
  }
  
  csv_files <- list.files(path = dir_path, pattern = paste(prefix, ".*\\.csv$", sep=""))
  
  # This will be the graph variable
  final_df = NULL;
  
  proc_nr = 0; 
  
  for (file in csv_files) {
    current <- read.csv(file.path(dir_path, file));
    current <- add_cumulative_columns(current = current, col_names = c("tx_kbps", "rx_kbps"))
    proc_nr <- proc_nr + 1;
    
    # Normalize the CPU usage (by the number of CPUs)
    current$tx_kbps <- current$tx_kbps * data_multiplier;
    current$rx_kbps <- current$rx_kbps * data_multiplier;
    current$rx_kbps_accumulated <- current$rx_kbps_accumulated * data_multiplier;
    current$tx_kbps_accumulated <- current$tx_kbps_accumulated * data_multiplier;
    current <- select(current, c(join_cols, c("tx_kbps", "rx_kbps", "tx_kbps_accumulated", "rx_kbps_accumulated")));
    
    # Suffix the last two column names
    colnames(current)[3] <- paste(colnames(current)[3], proc_nr, sep = "_proc_")
    colnames(current)[4] <- paste(colnames(current)[4], proc_nr, sep = "_proc_")
    colnames(current)[5] <- paste(colnames(current)[5], proc_nr, sep = "_proc_")
    colnames(current)[6] <- paste(colnames(current)[6], proc_nr, sep = "_proc_")
    
    if (is.null(final_df)) {
      final_df <- current;
    } else {
      final_df <- merge(final_df, current, by = join_cols);
    }
  }
  
  final_df$timestamp <- as.numeric(as.POSIXct(paste(final_df$date, final_df$time, sep = '-'), format="%Y-%m-%d-%H:%M:%S"));
  final_df$timestamp <- final_df$timestamp - min(final_df$timestamp);
  
  final_df_colnames <- colnames(final_df);
  
  tx_df <- select(final_df, c(c("timestamp"), final_df_colnames[grepl("^tx_kbps", final_df_colnames)]))
  rx_df <- select(final_df, c(c("timestamp"), final_df_colnames[grepl("^rx_kbps", final_df_colnames)]))
  
  new_col_names <- c()
  
  if (has_ps) {
    j <- 1
    
    for (i in 1:length(csv_files)) {
      if (csv_files[i] == ps_name) {
        if (!isTRUE(collocated_ps)) {
          new_col_names <- c(new_col_names, "Parameter Server", "Parameter Server Accumulated")
        } else {
          new_col_names <- c(new_col_names, paste("Parameter Server / Worker", j, sep = " "), paste("Parameter Server / Worker",  j, "Accumulated", sep = " "))
          j <- j + 1
        }
      } else {
        new_col_names <- c(new_col_names, paste("Worker", j, sep = " "), paste("Worker", j, "accumulated", sep = " "));
        j <- j + 1
      }
    }
  } else {
    for (i in 1:length(csv_files)) {
      new_col_names <- c(new_col_names, paste("Worker", i, sep = " "), paste("Worker", i, "accumulated", sep = " "));
    } 
  }

  new_col_names <- c("timestamp", new_col_names);
  colnames(tx_df) <- new_col_names; 
  colnames(rx_df) <- new_col_names;
  
  tx_data <- melt(tx_df, id = "timestamp");
  rx_data <- melt(rx_df, id = "timestamp");
  
  min_timestamp = 0;
  max_timestamp = max(tx_data$timestamp)
  
  final_tx <- ggplot(tx_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_tx <- final_tx + labs(title = paste(title_prefix, 'Per Process Sent Data with Accumulated Timeseries'), x = "Elapsed Time [s]", y = "Kilobits");
  final_tx <- final_tx + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_tx <- final_tx + labs(color = "Legend"); 
  
  final_rx <- ggplot(rx_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_rx <- final_rx + labs(title = paste(title_prefix, 'Per Process Received Data with Accumulated Timeseries'), x = "Elapsed Time [s]", y = "Kilobits");
  final_rx <- final_rx + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_rx <- final_rx + labs(color = "Legend");
  
  final_path <- file.path(save_path, dir_name);
  
  ggsave(plot = final_tx, filename = file.path(final_path, 'proc_tx_traffic_w_accumulated.png'), width = 14);
  ggsave(plot = final_rx, filename = file.path(final_path, 'proc_rx_traffic_w_accumulated.png'), width = 14);
}

# This function plots the data without the accumulated timeseries
plot_multi_line_timeseries <- function(dir_path, data_multiplier, save_path, dir_name, join_cols, has_ps, ps_name, collocated_ps, title_prefix, prefix) {
  if (missing(data_multiplier)) {
    #  1.0 means kbyte, 1024.0 is byte, etc. 
    data_multiplier = 1.0; 
  }
  
  if (missing(save_path)) {
    save_path = '/data/files/university/msc_thesis/2nd_semester/r_images/network';
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
    has_ps = FALSE;
    warning("No PS specified, assuming no PS is used");
  } else if (has_ps && missing(ps_name)) {
    stop("Must specify the name of the PS CSV");
  }
  
  if(missing(collocated_ps)) {
    collocated_ps <- FALSE;
  }
  
  if (missing(title_prefix)) {
    title_prefix = '';
  } else {
    title_prefix = paste(title_prefix, " - ");
  }
  
  if (missing(prefix)) {
    prefix = "network_traffic"
  }
  
  csv_files <- list.files(path = dir_path, pattern = paste(prefix, ".*\\.csv$", sep = ""))
  
  # This will be the graph variable
  final_df = NULL;
  
  proc_nr = 0; 
  
  for (file in csv_files) {
    current <- read.csv(file.path(dir_path, file));
    current <- add_cumulative_columns(current = current, col_names = c("tx_kbps", "rx_kbps"))
    proc_nr <- proc_nr + 1;
    
    # Normalize the CPU usage (by the number of CPUs)
    current$tx_kbps <- current$tx_kbps * data_multiplier;
    current$rx_kbps <- current$rx_kbps * data_multiplier;
    current <- select(current, c(join_cols, c("tx_kbps", "rx_kbps")));
    
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
  
  tx_df <- select(final_df, c(c("timestamp"), final_df_colnames[grepl("^tx_kbps", final_df_colnames)]))
  rx_df <- select(final_df, c(c("timestamp"), final_df_colnames[grepl("^rx_kbps", final_df_colnames)]))
  
  new_col_names <- c()
  
  if (has_ps) {
    j <- 1
    
    for (i in 1:length(csv_files)) {
      if (csv_files[i] == ps_name) {
        if (!isTRUE(collocated_ps)) {
          new_col_names <- c(new_col_names, "Parameter Server")
        } else {
          new_col_names <- c(new_col_names, paste("Parameter Server / Worker", j, sep = " "))
          j <- j + 1
        }
      } else {
        new_col_names <- c(new_col_names, paste("Worker", j, sep = " "));
        j <- j + 1
      }
    }
  } else {
    for (i in 1:length(csv_files)) {
      new_col_names <- c(new_col_names, paste("Worker", i, sep = " "));
    } 
  }
  
  new_col_names <- c("timestamp", new_col_names);
  colnames(tx_df) <- new_col_names; 
  colnames(rx_df) <- new_col_names;
  
  tx_data <- melt(tx_df, id = "timestamp");
  rx_data <- melt(rx_df, id = "timestamp");
  
  min_timestamp = 0;
  max_timestamp = max(tx_data$timestamp)
  
  final_tx <- ggplot(tx_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_tx <- final_tx + labs(title = paste(title_prefix, 'Per Process Sent Data'), x = "Elapsed Time [s]", y = "Kilobits");
  final_tx <- final_tx + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_tx <- final_tx + labs(color = "Legend"); 
  
  final_rx <- ggplot(rx_data, aes(x = timestamp, y = value, color = variable, group = variable)) + geom_line();
  final_rx <- final_rx + labs(title = paste(title_prefix, 'Per Process Received Data'), x = "Elapsed Time [s]", y = "Kilobits");
  final_rx <- final_rx + scale_x_continuous(limits = c(min_timestamp, max_timestamp), breaks = seq(min_timestamp, max_timestamp, by = by_period));
  final_rx <- final_rx + labs(color = "Legend");
  
  final_path <- file.path(save_path, dir_name);
  ggsave(plot = final_tx, filename = file.path(final_path, 'proc_tx_traffic.png'), width = 14);
  ggsave(plot = final_rx, filename = file.path(final_path, 'proc_rx_traffic.png'), width = 14);
}


# Experiment parameters

# Ethernet: n8
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/ps/',
#   "ps_n9" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/ps_n9/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n8/dar_ring/"
# )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('network_traffic_2019-04-17-14-16-36_node053_0.csv'),
#   "ps" = c('network_traffic_2019-04-15-14-57-20_node053_0.csv'),
#   "ps_n9" = c('network_traffic_2019-04-15-14-57-50_node038_0.csv'),
#   "dr" = c('network_traffic_2019-04-17-14-20-12_node030_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-15-16-25-35_node053_0.csv'),
#   "dar_ring" = c(''),
#   "car" = c('')
# )

# Ethernet: n16
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/ps/',
#   "ps_n17" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/ps_n17/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/network/ethernet/n16/dar_ring/"
# )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('network_traffic_2019-04-17-14-26-55_node008_0.csv'),
#   "ps" = c('network_traffic_2019-04-15-17-01-20_node030_0.csv'),
#   "ps_n17" = c('network_traffic_2019-04-15-17-18-27_node030_0.csv'),
#   "dr" = c('network_traffic_2019-04-17-14-27-19_node038_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-15-17-22-18_node008_0.csv'),
#   "dar_ring" = c(''),
#   "car" = c('')
# )


# Infiniband: n8
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/ps/',
#   "ps_n9" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/ps_n9/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n8/dar_ring/"
# )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('network_traffic_2019-04-17-15-39-55_10.149.0.53_0.csv'),
#   "ps" = c('network_traffic_2019-04-16-15-39-00_10.149.0.8_0.csv'),
#   "ps_n9" = c('network_traffic_2019-04-16-15-39-10_10.149.0.5_0.csv'),
#   "dr" = c('network_traffic_2019-04-17-15-40-05_10.149.0.30_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-16-15-39-25_10.149.0.39_0.csv'),
#   "dar_ring" = c(''),
#   "car" = c('')
# )

# Infiniband: n16
# csv_location = switch(
#   experiment,
#   "cps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/cps/',
#   "ps" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/ps/',
#   "ps_n17" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/ps_n17/',
#   "dr" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/dr',
#   "car" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/car',
#   "dar_pscpu" = '/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/dar_pscpu/',
#   "dar_ring" = "/data/files/university/msc_thesis/2nd_semester/r_images/network/infiniband/n16/dar_ring/"
# )
# 
# skip_files = switch(
#   experiment,
#   "cps" = c('network_traffic_2019-04-17-15-42-32_10.149.0.8_0.csv'),
#   "ps" = c('network_traffic_2019-04-16-16-29-12_10.149.0.30_0.csv'),
#   "ps_n17" = c('network_traffic_2019-04-16-16-35-01_10.149.0.2_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-16-16-45-40_10.149.0.8_0.csv'),
#   "dr" = c('network_traffic_2019-04-17-15-42-21_10.149.0.38_0.csv'),
#   "dar_ring" = c(''),
#   "car" = c('')
# )


# Ethernet: 4 Nodes
# skip_files = switch(
#   experiment,
#   "car" = c(''),
#   "cps" = c('network_traffic_2019-04-18-16-05-51_node055_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-18-16-06-00_node008_0.csv'),
#   "dar_ring" = c('network_traffic_2019-04-18-16-06-11_node044_0.csv'),
#   "dr" = c('network_traffic_2019-04-18-16-06-19_node031_0.csv'),
#   "ps" = c('network_traffic_2019-04-18-16-06-32_node035_0.csv'),
#   "ps_n5" = c('network_traffic_2019-04-18-16-06-47_node012_0.csv')
# )

# Ethernet: 12 Nodes
# skip_files = switch(
#   experiment,
#   "car" = c(''),
#   "cps" = c('network_traffic_2019-04-18-17-11-02_node008_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-18-17-11-07_node002_0.csv'),
#   "dar_ring" = c('network_traffic_2019-04-18-17-11-14_node020_0.csv'),
#   "dr" = c('network_traffic_2019-04-19-16-47-09_node031_0.csv'),
#   "ps" = c('network_traffic_2019-04-19-16-47-16_node008_0.csv'),
#   "ps_n13" = c('network_traffic_2019-04-19-16-47-32_node001_0.csv')
# )

# Infiniband: 4 Nodes
# skip_files = switch(
#   experiment,
#   "car" = c(''),
#   "cps" = c('network_traffic_2019-04-18-16-46-32_10.149.0.55_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-18-16-46-40_10.149.0.8_0.csv'),
#   "dar_ring" = c('network_traffic_2019-04-18-16-46-46_10.149.0.43_0.csv'),
#   "dr" = c('network_traffic_2019-04-18-16-46-53_10.149.0.31_0.csv'),
#   "ps" = c('network_traffic_2019-04-18-16-47-03_10.149.0.35_0.csv'),
#   "ps_n5" = c('network_traffic_2019-04-18-16-47-33_10.149.0.12_0.csv')
# )

# Infiniband: 12 Nodes
# skip_files = switch(
#   experiment,
#   "car" = c(''),
#   "cps" = c('network_traffic_2019-04-19-17-08-38_10.149.0.8_0.csv'),
#   "dar_pscpu" = c('network_traffic_2019-04-19-17-08-50_10.149.0.20_0.csv'),
#   "dar_ring" = c('network_traffic_2019-04-19-17-38-40_10.149.0.31_0.csv'),
#   "dr" = c('network_traffic_2019-04-19-17-38-46_10.149.0.8_0.csv'),
#   "ps" = c('network_traffic_2019-04-19-17-38-53_10.149.0.20_0.csv'),
#   "ps_n13" = c('network_traffic_2019-04-19-17-52-34_10.149.0.8_0.csv')
# )

# Caffe2 
skip_files = c('')


# Average across workers
upgraded_dataset <- add_cumulative_columns(current = average_datasets(dir_path = csv_location, excluded_files = skip_files))
plot_tx_rx(data = upgraded_dataset, save_path = csv_location, save_dir_name = plot_directory, title_prefix = title_prefix)

# Per process plot (multiple time series)
plot_multi_line_timeseries(dir_path = csv_location, save_path = csv_location, dir_name = plot_directory, has_ps = has_ps, ps_name = skip_files[1], collocated_ps = collocated_ps ,title_prefix = title_prefix)
plot_multi_line_timeseries_accumulated(dir_path = csv_location, save_path = csv_location, dir_name = plot_directory, has_ps = has_ps, ps_name = skip_files[1], collocated_ps = collocated_ps, title_prefix = title_prefix)

