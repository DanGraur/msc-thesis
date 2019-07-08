library("reshape2")
library("ggplot2")
library("dplyr")

### Invariants

# Note: I inverted the 'Non-collocated Parameter Server' with its extra-node counterpart. 
strategies = c(
  "Caffe2", 
  "Collective Allreduce", 
  "Collocated Parameter Server", 
  "Distributed Allreduce w/ PSCPU", 
  "Distributed Allreduce w/ Ringx1",
  "Distributed Replicated",
  "Non-collocated Parameter Server w/ Extra Node",
  "Non-collocated Parameter Server"
)

out_img_prefix = c(
  "c2", 
  "car", 
  "cps", 
  "dar_pscpu", 
  "dar_ring",
  "dr",
  "ps",
  "ps_en"
)

rx_folder_name = "rx"
tx_folder_name = "tx"

# Given a directory which hosts, CSVs, this function will aggregate the CSVs together, store the resulting CSV, and return the max on y and x
aggregate_architecture <- function(dir_path, has_ps, ps_csv_name, collocated_ps, save_path, architecture_prefix) {
  x_max = 0
  y_max_tx = 0
  y_max_rx = 0
  
  ################################# START - Initial function checks - START #################################
  
  if (missing(save_path)) {
    stop("Must mention a place where the plots will be saved")
  } 
  
  if (missing(has_ps)) {
    has_ps = FALSE;
    warning("No PS specified, assuming no PS is used");
  } else if (has_ps && missing(ps_csv_name)) {
    stop("Must specify the name of the PS CSV");
  }
  
  if(missing(collocated_ps)) {
    collocated_ps <- FALSE;
  }
  
  #  1.0 means kbyte, 1024.0 is byte, etc. 
  data_multiplier = 1.0; 
  
  join_cols <- c("date", "time");
  
  ################################# END - Initial function checks - END #################################
  
  csv_files <- list.files(path = dir_path, pattern = "network_traffic.*\\.csv$")
  
  # This will be the graph variable
  final_df = NULL;
  
  proc_nr = 0; 
  
  for (file in csv_files) {
    current <- read.csv(file.path(dir_path, file));
    proc_nr <- proc_nr + 1;
    
    # Normalize the CPU usage (by the number of CPUs)
    current$tx_kbps <- current$tx_kbps * data_multiplier;
    current$rx_kbps <- current$rx_kbps * data_multiplier;
    current <- select(current, c(join_cols, c("tx_kbps", "rx_kbps")));
    
    # Find the max tx and rx 
    y_max_tx = max(y_max_tx, current$tx_kbps)
    y_max_rx = max(y_max_rx, current$rx_kbps)
    
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
      if (csv_files[i] == ps_csv_name) {
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
  
  # We assume that we have the following structure:
  #     averages:
  #           - tx:
  #               <tx_files>*
  #           - rx: 
  #               <rx_files>*
  write.csv(tx_df, file.path(save_path, tx_folder_name, paste(architecture_prefix, "tx_per_proc.csv", sep="_")), row.names = FALSE)
  write.csv(rx_df, file.path(save_path, rx_folder_name, paste(architecture_prefix, "rx_per_proc.csv", sep="_")), row.names = FALSE)

  returnValue(c(max(tx_df$timestamp), y_max_tx, y_max_rx))  
}

# This is a function which creates a ggplot like pallette
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}


# This function plots the CSV passed to it
plot_multi_line_timeseries <- function(csv_path, title_prefix, title, out_img_path, max_x, max_y) {
  
  if (missing(out_img_path)) {
    stop("Must mention a place where the plots will be saved")
  } 
  
  print(csv_path)
  
  df <- read.csv(csv_path)
  timeseries_csv <- melt(df, id = "timestamp");
  # show(timeseries_csv)
  
  # This is for the worker / parameter server case
  nice_col_names = gsub("\\.{3}", " / ", colnames(df)[2:length(colnames(df))], perl = FALSE)
  # This is for the general case
  nice_col_names = gsub("\\.", " ", nice_col_names, perl = FALSE)
  
  final_plot <- ggplot(timeseries_csv, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, title, sep = " - "), x = "Elapsed Time [s]", y = "Network Traffic [Kbits]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, max_x), breaks = seq(0, max_x, by = 40))
  final_plot <- final_plot + scale_y_continuous(limits = c(0, max_y), breaks = seq(0, max_y, by = 10000)) #  10000 for ethernet; 100000 for InfiniBand
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = nice_col_names, values = gg_color_hue(length(nice_col_names))) 
  # final_plot <- final_plot + scale_color_manual(labels = c("CPU Utilization"), values = c("blue")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  ggsave(plot = final_plot, filename = out_img_path, width = 14)
}





# Experiment parameters
skip_file_chooser <- function(node_count, ethernet) {
  returned_value = NULL
  
  if (ethernet) {
    # Ethernet case
    returned_value = switch (as.character(node_count),
                             "4" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-18-16-05-51_node055_0.csv',
                               dar_pscpu='network_traffic_2019-04-18-16-06-00_node008_0.csv',
                               dar_ring='network_traffic_2019-04-18-16-06-11_node043_0.csv',
                               dr='network_traffic_2019-04-18-16-06-19_node031_0.csv',
                               ps='network_traffic_2019-04-18-16-06-32_node035_0.csv',
                               ps_n5='network_traffic_2019-04-18-16-06-47_node012_0.csv'
                             ),
                             "8" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-17-14-16-36_node053_0.csv',
                               dar_pscpu='network_traffic_2019-04-15-16-25-35_node053_0.csv',
                               dar_ring='network_traffic_2019-04-15-16-26-01_node030_0.csv',
                               dr='network_traffic_2019-04-17-14-20-12_node030_0.csv',
                               ps='network_traffic_2019-04-15-14-57-20_node053_0.csv',
                               ps_n9='network_traffic_2019-04-15-14-57-50_node038_0.csv'
                             ),
                             "12" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-18-17-11-02_node008_0.csv',
                               dar_pscpu='network_traffic_2019-04-18-17-11-07_node002_0.csv',
                               dar_ring='network_traffic_2019-04-18-17-11-14_node020_0.csv',
                               dr='network_traffic_2019-04-19-16-47-09_node031_0.csv',
                               ps='network_traffic_2019-04-19-16-47-16_node008_0.csv',
                               ps_n13='network_traffic_2019-04-19-16-47-32_node001_0.csv'
                             ),
                             "16" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-17-14-26-55_node008_0.csv',
                               dar_pscpu='network_traffic_2019-04-15-17-22-18_node008_0.csv',
                               dar_ring='network_traffic_2019-04-16-13-40-33_node030_0.csv',
                               dr='network_traffic_2019-04-17-14-27-19_node038_0.csv',
                               ps='network_traffic_2019-04-15-17-01-20_node030_0.csv',
                               ps_n17='network_traffic_2019-04-15-17-18-27_node030_0.csv'
                             )
    )
  } else {
    # Infiniband case 
    returned_value = switch (as.character(node_count),
                             "4" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-18-16-46-32_10.149.0.55_0.csv',
                               dar_pscpu='network_traffic_2019-04-18-16-46-40_10.149.0.8_0.csv',
                               dar_ring='network_traffic_2019-04-18-16-46-46_10.149.0.43_0.csv',
                               dr='network_traffic_2019-04-18-16-46-53_10.149.0.31_0.csv',
                               ps='network_traffic_2019-04-18-16-47-03_10.149.0.35_0.csv',
                               ps_n5='network_traffic_2019-04-18-16-47-33_10.149.0.12_0.csv'
                             ),
                             "8" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-17-15-39-55_10.149.0.53_0.csv',
                               dar_pscpu='network_traffic_2019-04-16-15-39-25_10.149.0.39_0.csv',
                               dar_ring='network_traffic_2019-04-16-15-54-02_10.149.0.30_0.csv',
                               dr='network_traffic_2019-04-17-15-40-05_10.149.0.30_0.csv',
                               ps='network_traffic_2019-04-16-15-39-00_10.149.0.8_0.csv',
                               ps_n9='network_traffic_2019-04-16-15-39-10_10.149.0.5_0.csv'
                             ),
                             "12" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-19-17-08-38_10.149.0.8_0.csv',
                               dar_pscpu='network_traffic_2019-04-19-17-08-50_10.149.0.20_0.csv',
                               dar_ring='network_traffic_2019-04-19-17-38-40_10.149.0.31_0.csv',
                               dr='network_traffic_2019-04-19-17-38-46_10.149.0.8_0.csv',
                               ps='network_traffic_2019-04-19-17-38-53_10.149.0.20_0.csv',
                               ps_n13='network_traffic_2019-04-19-17-52-34_10.149.0.8_0.csv'
                             ),
                             "16" = c(
                               c2='',
                               car='',
                               cps='network_traffic_2019-04-17-15-42-32_10.149.0.8_0.csv',
                               dar_pscpu='network_traffic_2019-04-16-16-45-40_10.149.0.8_0.csv',
                               dar_ring='network_traffic_2019-04-16-16-45-58_10.149.0.3_0.csv',
                               dr='network_traffic_2019-04-17-15-42-21_10.149.0.38_0.csv',
                               ps='network_traffic_2019-04-16-16-29-12_10.149.0.30_0.csv',
                               ps_n17='network_traffic_2019-04-16-16-35-01_10.149.0.2_0.csv'
                             )
    )
  }
  
  returnValue(returned_value)
}


parameter_server_info_producer <- function(experiment) {
  has_ps = switch(
    experiment,
    "car" = FALSE,
    "c2" = FALSE,
    TRUE
  )
  
  collocated_ps = switch (
    experiment,
    "cps" = TRUE,
    "dr" = TRUE,
    "dar_pscpu" = TRUE,
    "dar_ring" = TRUE,
    FALSE
  )

  returnValue(c(has_ps, collocated_ps))    
}


# This is where the actual code is; above are only method declarations

ethernet = FALSE
node_count = 16

# TODO: divide the y lims by as much as needed to zoom in on the workers

connection_type = 'ethernet'
plot_connection_type = "Ethernet"

if (!ethernet) {
  connection_type = 'infiniband'   
  plot_connection_type = "InfiniBand"
}

# Location of the top dir where the results are stored
top_dir_path = paste('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/', connection_type, '/n', node_count, sep = "")

# Where the averaged CSVs will be stored
save_dir = file.path(top_dir_path, "averages_network")
tx_dir = file.path(save_dir, tx_folder_name)
rx_dir = file.path(save_dir, rx_folder_name)

dir.create(save_dir, showWarnings = FALSE)
dir.create(tx_dir, showWarnings = FALSE)
dir.create(rx_dir, showWarnings = FALSE)

skip_files = skip_file_chooser(node_count = node_count, ethernet = ethernet)

master_max_x = 0
master_max_y_tx = 0
master_max_y_rx = 0

# Actuially compute and save the CSVs
for (dir_path in list.dirs(top_dir_path, recursive = FALSE)) {
  architecture = basename(dir_path)
  
  if (architecture == "averages" || architecture == "averages_network") {
    next
  }
  
  temp = parameter_server_info_producer(architecture)
  
  temp = aggregate_architecture(
    dir_path = dir_path,
    has_ps = temp[1],
    ps_csv_name = skip_files[architecture],
    temp[2],
    save_dir,
    architecture
  )
  
  master_max_x = max(master_max_x, temp[1])
  master_max_y_tx = max(master_max_y_tx, temp[2])
  master_max_y_rx = max(master_max_y_rx, temp[3])
}

# This is for scaling down the upper limit, so the worker patterns are visible
# master_max_y_rx = master_max_y_rx / 1000000
# master_max_y_tx = master_max_y_tx / 1000000

# Not it's time to plot the graphs
idx = 1
for (csv_path in list.files(pattern = "\\.csv", path = tx_dir, full.names = TRUE)) {
  architecture = out_img_prefix[idx]
  temp = parameter_server_info_producer(architecture)
  
  plot_multi_line_timeseries(
    csv_path = csv_path,
    title_prefix = paste('(', plot_connection_type, ') ', strategies[idx], sep = ""),
    "Transmitted Kbits",
    sub("\\.csv", "\\.png", csv_path),
    master_max_x,
    master_max_y_tx
  )
  
  idx = idx + 1
}

idx = 1
for (csv_path in list.files(pattern = "\\.csv", path = rx_dir, full.names = TRUE)) {
  architecture = out_img_prefix[idx]
  temp = parameter_server_info_producer(architecture)
  
  plot_multi_line_timeseries(
    csv_path = csv_path,
    title_prefix = paste('(', plot_connection_type, ') ', strategies[idx], sep = ""),
    "Received Kbits",
    sub("\\.csv", "\\.png", csv_path),
    master_max_x,
    master_max_y_tx
  )
  
  idx = idx + 1
}


# Example

# aggregate_architecture(
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n12/c2',
#   has_ps,
#   skip_files[experiment],
#   collocated_ps,
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n12/averages_netwrok',
#   experiment
# )
# 
# 
# # (csv_path, title_prefix, title, out_img_path, max_x, max_y)
# plot_multi_line_timeseries(
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n12/averages_netwrok/tx/c2_tx_per_proc.csv',
#   strategies[3],
#   "Transmitted Kbits",
#   "/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n12/averages_netwrok/tx/c2_tx_per_proc.png",
#   1000,
#   130000
# )
# 
# plot_multi_line_timeseries(
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n12/averages_netwrok/rx/c2_rx_per_proc.csv',
#   strategies[3],
#   "Received Kbits",
#   "/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n12/averages_netwrok/rx/c2_rx_per_proc.png",
#   1000,
#   130000
# )

