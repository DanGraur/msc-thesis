library("reshape2")
library("ggplot2")
library("dplyr")

# Invariants - START
# The DAR Ring has a controller process, which actually acts as a Parameter Server of sorts

# Ethernet - 8 Nodes
# skip_files = c(
#   c2='',
#   car='',
#   cps='cpu_mem_eval_2019-04-10-12-59-10_node053_0.csv',
#   dar_pscpu='',
#   dar_ring='cpu_mem_eval_2019-04-10-14-48-02_node053_0.csv',
#   dr='cpu_mem_eval_2019-04-10-14-15-45_node053_0.csv',
#   ps='cpu_mem_eval_2019-04-10-13-04-57_node030_0.csv',
#   ps_n9='cpu_mem_eval_2019-04-11-13-09-54_node053_0.csv'
# )

# csv_files = c(
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/c2 _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/car _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/cps _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dar_pscpu _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dar_ring _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dr _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/ps _cpu_mem_avg.csv',
#   '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/ps_n9 _cpu_mem_avg.csv'
# )

strategies = c(
  "Caffe2", 
  "Collective Allreduce", 
  "Collocated Parameter Server", 
  "Distributed Allreduce w/ PSCPU", 
  "Distributed Allreduce w/ Ringx1",
  "Distributed Replicated",
  "Non-collocated Parameter Server",
  "Non-collocated Parameter Server w/ Extra Node"
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

# Invariants - END

# This function aggragates the mem_footprint and CPU usage into a single DF
  average_datasets <- function(dir_path, join_cols, excluded_file, prefix_save_file, save_dir) {
  if (missing(dir_path)) {
    stop("Must provide a dir path");
  }
  
  if (missing(join_cols)) {
    join_cols <- c("date", "time");
  }
  
  if (missing(excluded_file)) {
    excluded_file = '';
  }
  
  if (missing(prefix_save_file)) {
    prefix_save_file = ''
  }
  
  csv_files <- list.files(path = dir_path, pattern = "cpu_mem_eval.*\\.csv$")
  
  final = NULL;
  
  for (file in csv_files) {
    
    if (file == excluded_file) {
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
  
  # Noralize the CPU percentage (32 cores ... 100%)
  cpu_norm_factor = 32.0 * 100.0;
  # Denormalize the Mem footprint (64 GB ... 100%)
  mem_denorm_factor = 64.0 / 100.0;
  # Number of employed CSVs
  mean_denominator = length(csv_files)
  if (excluded_file != '') {
    mean_denominator = mean_denominator - 1
  }
  
  final$proc_perc <- (final$proc_perc / length(csv_files)) / cpu_norm_factor;
  final$mem_perc <- (final$mem_perc / length(csv_files)) * mem_denorm_factor;
  
  # Create a timestamp
  final$timestamp <- as.numeric(as.POSIXct(paste(final$date, final$time, sep = '-'), format="%Y-%m-%d-%H:%M:%S"));
  final$timestamp <- final$timestamp - min(final$timestamp);
  
  # Discard unecessary columns
  final = select(final, c("timestamp", "proc_perc", "mem_perc"))
  colnames(final) = c('timestamp', "proc_perc", "mem_gb")
  
  # Save the CSV in the same folder as the other CSVs
  write.csv(x = final, file = file.path(save_dir, paste(prefix_save_file, "cpu_mem_avg.csv", sep = "")))
  return(final);
}



# Gets the max of the x axis and y axis 
get_x_and_y_max <- function(timeseries_dfs) {
  max_x = 0
  max_y_cpu = 0
  max_y_mem = 0
  
  for (df_path in timeseries_dfs) {
    df = read.csv(df_path)
    max_x = max(max_x, max(df$timestamp))
    max_y_cpu = max(max_y_cpu, max(df$proc_perc))
    max_y_mem = max(max_y_mem, max(df$mem_gb))
  }
  
  returnValue(c(max_x, max_y_cpu, max_y_mem))
}





# This function plots the memory and CPU from an aggregated DF
plot_multi_line_timeseries <- function(df_path, title_prefix, out_img_path, x_max, y_max_mem) {
  df = read.csv(df_path)
  
  # Melt the CSV to a plottable format
  timeseries_csv = melt(df,  id.vars = "timestamp", measure.vars = c("proc_perc", "mem_gb"))
  print(timeseries_csv)
  
  
  # Plot formatting
  final_plot <- ggplot(timeseries_csv, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, "CPU Utilization and Memory Footprint", sep = " - "), x = "Elapsed Time [s]", y = "CPU Utilization [%]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, x_max), breaks = seq(0, x_max, by = 40))
  final_plot <- final_plot + scale_y_continuous(sec.axis = sec_axis(~.*y_max_mem, name = "Memory Footprint [GB]"))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = c("CPU Utilization", "Memory Footprint"), values = c("blue", "red")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  # show(final_plot)
  ggsave(plot = final_plot, filename = out_img_path, width = 14)
}  



# This function plots the memory and CPU for each of the tested frameworks, but not in the same file
plot_cpu_and_mem_separate <- function(df_path, title_prefix, out_img_path_cpu, out_img_path_mem, x_max, y_max) {
  df = read.csv(df_path)
  
  # Melt the CSV to a plottable format
  timeseries_csv_cpu = melt(df,  id.vars = "timestamp", measure.vars = c("proc_perc"))
  timeseries_csv_mem = melt(df,  id.vars = "timestamp", measure.vars = c("mem_gb"))
  
  # CPU - Plot formatting 
  final_plot <- ggplot(timeseries_csv_cpu, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, "CPU Utilization", sep = " - "), x = "Elapsed Time [s]", y = "CPU Utilization [%]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, x_max), breaks = seq(0, x_max, by = 40))
  final_plot <- final_plot + scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1.0, by = 0.05))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = c("CPU Utilization"), values = c("blue")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  ggsave(plot = final_plot, filename = out_img_path_cpu, width = 14)
  
  # MEM - Plot formatting 
  final_plot <- ggplot(timeseries_csv_mem, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, "Memory Footprint", sep = " - "), x = "Elapsed Time [s]", y = "Memory Footprint [GB]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, x_max), breaks = seq(0, x_max, by = 40))
  final_plot <- final_plot + scale_y_continuous(limits = c(0, y_max), breaks = seq(0, y_max, by = 1))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = c("Memory Footprint"), values = c("red")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  ggsave(plot = final_plot, filename = out_img_path_mem, width = 14)
}





# This function plots the memory and CPU from an aggregated DF
plot_multi_line_timeseries <- function(df_path, title_prefix, out_img_path, x_max, y_max_mem) {
  df = read.csv(df_path)
  
  # Melt the CSV to a plottable format
  timeseries_csv = melt(df,  id.vars = "timestamp", measure.vars = c("proc_perc", "mem_gb"))
  print(timeseries_csv)
  
  
  # Plot formatting
  final_plot <- ggplot(timeseries_csv, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, "CPU Utilization and Memory Footprint", sep = " - "), x = "Elapsed Time [s]", y = "CPU Utilization [%]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, x_max), breaks = seq(0, x_max, by = 40))
  final_plot <- final_plot + scale_y_continuous(sec.axis = sec_axis(~.*y_max_mem, name = "Memory Footprint [GB]"))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = c("CPU Utilization", "Memory Footprint"), values = c("blue", "red")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  # show(final_plot)
  ggsave(plot = final_plot, filename = out_img_path, width = 14)
}  





# This function plots the memory  and CPU for each of the tested frameworks of two clusters comparatively. The CPU and MEM are plotted to different files
plot_cpu_and_mem_separate_vs <- function(df_path_1, df_path_2, title_prefix, out_img_path_cpu, out_img_path_mem, x_max, y_max, cluster_sizes) {
  print(df_path_1)
  print(df_path_2)
  
  df_1 = read.csv(df_path_1)
  df_2 = read.csv(df_path_2)
  
  # Melt the CSV to a plottable format
  df_1 = select(df_1, c("timestamp", "proc_perc", "mem_gb"))
  df_2 = select(df_2, c("timestamp", "proc_perc", "mem_gb"))
  df = merge(df_1, df_2, by = c("timestamp"), all = TRUE)
  colnames(df) = c("timestamp", "proc_perc_1", "mem_gb_1", "proc_perc_2", "mem_gb_2")
  df_1 = select(df, c("timestamp", "proc_perc_1", "proc_perc_2"))  
  df_2 = select(df, c("timestamp", "mem_gb_1", "mem_gb_2"))
  
  timeseries_csv_cpu = melt(df_1,  id.vars = "timestamp")
  timeseries_csv_mem = melt(df_2,  id.vars = "timestamp")
  
  # CPU - Plot formatting 
  final_plot <- ggplot(timeseries_csv_cpu, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, "CPU Utilization", sep = " - "), x = "Elapsed Time [s]", y = "CPU Utilization [%]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, x_max), breaks = seq(0, x_max, by = 40))
  final_plot <- final_plot + scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1.0, by = 0.05))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = cluster_sizes, values = c("blue", "red")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  ggsave(plot = final_plot, filename = out_img_path_cpu, width = 14)
  
  # MEM - Plot formatting 
  final_plot <- ggplot(timeseries_csv_mem, aes_string(x = "timestamp", y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, "Memory Footprint", sep = " - "), x = "Elapsed Time [s]", y = "Memory Footprint [GB]")
  final_plot <- final_plot + scale_x_continuous(limits = c(0, x_max), breaks = seq(0, x_max, by = 40))
  final_plot <- final_plot + scale_y_continuous(limits = c(0, y_max), breaks = seq(0, y_max, by = 1))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = cluster_sizes, values = c("blue", "red")) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  ggsave(plot = final_plot, filename = out_img_path_mem, width = 14)
}



# This function plots merged timeseries graphs of all the architectures in a dir  


plot_all_timeseries <- function(df_paths, connection_type, prefix, out_img_prefix) {
  maxes = get_x_and_y_max(df_paths)
  x_max = maxes[1]
  y_max = maxes[3]
  idx = 1
  
  for (df_path in df_paths) {
    plot_multi_line_timeseries(
      df_path,
      paste('(', connection_type, ') ', prefix[idx]),
      file.path(dirname(df_path), paste(out_img_prefix[idx], "cpu_and_mem_coplot.png", sep = "_")),
      x_max,
      y_max
    )
    idx = idx + 1
  }
  
}



# This function plots the CPU and MEM graphs for all the aggregated DFs in a versus fashion
plot_all_timeseries_separate_vs <- function(df_paths, connection_type, prefix, out_img_prefix, cluster_sizes) {
  maxes = get_x_and_y_max(df_paths)
  x_max = maxes[1]
  y_max = maxes[3]
  idx = 1
  
  save_path = file.path(dirname(df_paths[1]), "infini_vs_ethernet_8n")
  dir.create(save_path)
  
  half_length = as.integer(length(df_paths) / 2) - 1
  
  for (df_idx in 0:half_length) {
    plot_cpu_and_mem_separate_vs(
      df_paths[[2 * df_idx + 1]],
      df_paths[[2 * df_idx + 2]],
      paste('(', connection_type, ') ', prefix[idx], sep = ""),
      file.path(save_path, paste(out_img_prefix[idx], "cpu_plot_comparative.png", sep = "_")),
      file.path(save_path, paste(out_img_prefix[idx], "mem_plot_comparative.png", sep = "_")),
      x_max,
      y_max,
      cluster_sizes = cluster_sizes
    )
    idx = idx + 1
  }
  
}


# This function plots the CPU and MEM in a comparative fashion for all the matching DFs
plot_all_timeseries_separate <- function(df_paths, connection_type, prefix, out_img_prefix) {
  maxes = get_x_and_y_max(df_paths)
  x_max = maxes[1]
  y_max = maxes[3]
  idx = 1
  
  paired_csvs = list()
  seen = c()
  
  for (df_path in df_paths) {
    plot_cpu_and_mem_separate(
      df_path,
      paste('(', connection_type, ') ', prefix[idx], sep = ""),
      file.path(dirname(df_path), paste(out_img_prefix[idx], "cpu_plot.png", sep = "_")),
      file.path(dirname(df_path), paste(out_img_prefix[idx], "mem_plot.png", sep = "_")),
      x_max,
      y_max
    )
    idx = idx + 1
  }
  
}

### This method is used for choosing the value for skipping files array

skip_file_chooser <- function(node_count, ethernet) {
  returned_value = NULL
  
  if (ethernet) {
    # EThernet case
    returned_value = switch (as.character(node_count),
      "4" = c(
          c2='',
          car='',
          cps='cpu_mem_eval_2019-04-18-16-05-51_node055_0.csv',
          dar_pscpu='cpu_mem_eval_2019-04-18-16-06-00_node008_1.csv',
          dar_ring='cpu_mem_eval_2019-04-18-16-06-11_node043_0.csv',
          dr='cpu_mem_eval_2019-04-18-16-06-19_node031_1.csv',
          ps='cpu_mem_eval_2019-04-18-16-06-32_node035_0.csv',
          ps_n5='cpu_mem_eval_2019-04-18-16-06-47_node012_0.csv'
      ),
      "8" = c(
          c2='',
          car='',
          cps='cpu_mem_eval_2019-04-10-12-59-10_node053_0.csv',
          dar_pscpu='',
          dar_ring='cpu_mem_eval_2019-04-10-14-48-02_node053_0.csv',
          dr='cpu_mem_eval_2019-04-10-14-15-45_node053_0.csv',
          ps='cpu_mem_eval_2019-04-10-13-04-57_node030_0.csv',
          ps_n9='cpu_mem_eval_2019-04-11-13-09-54_node053_0.csv'
      ),
      "12" = c(
        c2='',
        car='',
        cps='cpu_mem_eval_2019-04-18-17-11-02_node008_1.csv',
        dar_pscpu='cpu_mem_eval_2019-04-18-17-11-07_node002_1.csv',
        dar_ring='cpu_mem_eval_2019-04-18-17-11-14_node020_0.csv',
        dr='cpu_mem_eval_2019-04-19-16-47-09_node031_1.csv',
        ps='cpu_mem_eval_2019-04-19-16-47-16_node008_0.csv',
        ps_n13='cpu_mem_eval_2019-04-19-16-47-32_node001_0.csv'
      ),
      "16" = c(
        c2='',
        car='',
        cps='cpu_mem_eval_2019-04-10-16-39-30_node008_0.csv',
        dar_pscpu='cpu_mem_eval_2019-04-10-15-46-45_node030_1.csv',
        dar_ring='cpu_mem_eval_2019-04-10-15-46-37_node008_0.csv',
        dr='cpu_mem_eval_2019-04-10-16-24-13_node008_1.csv',
        ps='cpu_mem_eval_2019-04-10-16-39-11_node030_0.csv',
        ps_n17='0_ps.csv',
        dr_forward='cpu_mem_eval_2019-07-06-17-11-37_node030_1.csv'
      )
    )
  } else {
    # Infiniband case 
    returned_value = switch (as.character(node_count),
      "4" = c(
        c2='',
        car='',
        cps='cpu_mem_eval_2019-04-18-16-46-32_10.149.0.55_1.csv',
        dar_pscpu='cpu_mem_eval_2019-04-18-16-46-40_10.149.0.8_0.csv',
        dar_ring='cpu_mem_eval_2019-04-18-16-46-46_10.149.0.43_1.csv',
        dr='cpu_mem_eval_2019-04-18-16-46-53_10.149.0.31_0.csv',
        ps='cpu_mem_eval_2019-04-18-16-47-03_10.149.0.35_0.csv',
        ps_n5='cpu_mem_eval_2019-04-18-16-47-33_10.149.0.12_0.csv'
      ),
      "8" = c(
        c2='',
        car='',
        cps='0_ps.csv',
        dar_pscpu='cpu_mem_eval_2019-04-14-21-50-32_10.149.0.30_1.csv',
        dar_ring='cpu_mem_eval_2019-04-14-21-51-16_10.149.0.38_1.csv',
        dr='0_ps.csv',
        ps='0_ps.csv',
        ps_n9='cpu_mem_eval_2019-04-14-21-37-39_10.149.0.53_0.csv'
      ),
      "12" = c(
        c2='',
        car='',
        cps='cpu_mem_eval_2019-04-19-17-08-38_10.149.0.8_1.csv',
        dar_pscpu='cpu_mem_eval_2019-04-19-17-08-50_10.149.0.20_1.csv',
        dar_ring='cpu_mem_eval_2019-04-19-17-38-40_10.149.0.31_0.csv',
        dr='cpu_mem_eval_2019-04-19-17-38-46_10.149.0.8_0.csv',
        ps='cpu_mem_eval_2019-04-19-17-38-53_10.149.0.20_0.csv',
        ps_n13='cpu_mem_eval_2019-04-19-17-52-34_10.149.0.8_0.csv'
      ),
      "16" = c(
        c2='',
        car='',
        cps='cpu_mem_eval_2019-04-14-23-05-40_10.149.0.30_0.csv',
        dar_pscpu='cpu_mem_eval_2019-04-14-22-06-08_10.149.0.8_0.csv',
        dar_ring='cpu_mem_eval_2019-04-14-22-06-21_10.149.0.46_0.csv',
        dr='cpu_mem_eval_2019-04-14-23-04-53_10.149.0.46_0.csv',
        ps='cpu_mem_eval_2019-04-14-23-05-20_10.149.0.8_0.csv',
        ps_n17='cpu_mem_eval_2019-04-14-23-43-06_10.149.0.8_0.csv',
        dr_forward='cpu_mem_eval_2019-04-14-23-04-53_10.149.0.46_0.csv'
      )
    )
  }
  
  returnValue(returned_value)
}

##
## 
##

# average_datasets(
#   dir_path = '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n1/c2',
#   join_cols = c("date", "time"),
#   excluded_file = '',
#   prefix_save_file = 'c2',
#   save_dir = '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n1/c2/backup'
# )
# 
# stop()

plot_cpu_and_mem_separate(
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n1/c2/c2_cpu_mem_avg.csv',
  "(Ethernet) Caffe2",
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n1/c2/caffe2_CPU_1node.png',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n1/c2/caffe2_MEM_1node.png',
  41,
  0.5
)

stop()

# CPU & MEM - N8 vs N16
csvs = c(
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/c2 _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/c2 _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/car _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/car _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/cps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/cps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dar_pscpu _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/dar_pscpu _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dar_ring _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/dar_ring _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dr _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/dr _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/ps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/ps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/ps_n9 _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n16/averages/ps_n17 _cpu_mem_avg.csv'
)

# CPU & MEM - Ethernet vs InfiniBand
csvs = c(
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/c2 _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/c2 _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/car _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/car _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/cps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/cps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dar_pscpu _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/dar_pscpu _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dar_ring _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/dar_ring _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/dr _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/dr _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/ps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/ps _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/ethernet/n8/averages/ps_n9 _cpu_mem_avg.csv',
  '/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/infiniband/n8/averages/ps_n9 _cpu_mem_avg.csv'
)


# TODO: Continue from here
average_datasets(dir_path, excluded_file = skip_files[architecture], prefix_save_file = paste(architecture, "_"), save_dir = save_dir)

stop()

plot_all_timeseries_separate_vs(
  df_paths = csvs,
  connection_type = "Ethernet & InfiniBand",
  prefix = strategies,
  out_img_prefix = out_img_prefix,
  c("Ethernet", "InfiniBand")
)

stop()

##
## The following lines of code will create the aforementioned dataset for all the folders in another topfolder
##

# for (ethernet in c(TRUE, FALSE)) {
#   for (node_count in c(4, 8, 12, 16)) {
    # connection_type = 'ethernet'
    # ethernet = FALSE
    # node_count = 16
    # 
    # if (!ethernet) {
    #   connection_type = 'infiniband'
    # }
    # 
    # plot_connection_type = "Ethernet"
    # 
    # if (!ethernet) {
    #   plot_connection_type = "InfiniBand"
    # }
    # 
    # # The node count and the skip files
    # skip_files = skip_file_chooser(node_count, ethernet)
    # 
    # # Location of the top dir where the results are stored
    # top_dir_path = paste('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/', connection_type, '/n', node_count, sep = "")
    # 
    # # Where the averaged CSVs will be stored
    # save_dir = file.path(top_dir_path, "averages")
    # dir.create(save_dir, showWarnings = FALSE)
    # 
    # # Actuially compute and save the CSVs
    # for (dir_path in list.dirs(top_dir_path, recursive = FALSE)) {
    #   architecture = basename(dir_path)
    #   if (architecture == "averages") {
    #     next
    #   }
    #   average_datasets(dir_path, excluded_file = skip_files[architecture], prefix_save_file = paste(architecture, "_"), save_dir = save_dir)
    # }
    # 
    # plot_all_timeseries_separate(
    #   list.files(path = save_dir, pattern = ".*\\.csv$", full.names = TRUE),
    #   plot_connection_type,
    #   strategies,
    #   out_img_prefix
    # )
#   }
# }

    
    

