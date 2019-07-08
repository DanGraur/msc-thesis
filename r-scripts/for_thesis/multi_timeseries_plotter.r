library("base")
library("dplyr")
library("ggplot2")
library("reshape2")


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

color_array <- c("blue", "red", "green", "yellow", "purple", "orange", "cyan", "brown", "darkgreen", "darkorchid", "gold", "lightblue")

plot_multi_line_timeseries <- function(timeseries_dfs, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
  
  # Filter the timeseries, select the relevant columns, and identify possible possitions for breaks in the plot
  timeseries_csv = NULL
  idx = 0
  
  # How many columns per file are there?
  columns_per_file = length(y_axis_cols) / length(timeseries_dfs)
  
  for (df in timeseries_dfs) {
    
    temp_idx = idx * columns_per_file
    
    select_column_names = c(x_axis_col)
    rename_column_names = c(x_axis_col)
    
    for (offset in 1:columns_per_file) {
      select_column_names = c(select_column_names, y_axis_cols[temp_idx + offset])
      rename_column_names = c(rename_column_names, paste(y_axis_cols[temp_idx + offset], idx, sep = "_"))
    }
    
    temp = select(df, select_column_names)
    colnames(temp) = rename_column_names
    idx = idx + 1
    
    if (is.null(timeseries_csv)) {
      timeseries_csv <- temp
    } else {
      timeseries_csv <- merge(timeseries_csv, temp, by = x_axis_col, all = TRUE)
    }
  }
  
  # Get the lower and upper bounds of the x axis, as well as the positions where the breaks occur
  lo_x_limit = min(timeseries_csv[x_axis_col])
  hi_x_limit = max(timeseries_csv[x_axis_col])
  breaks = timeseries_csv[x_axis_col]
  
  # Melt the CSV to a plottable format
  timeseries_csv = melt(timeseries_csv, id = x_axis_col)
  
  
  # Plot formatting
  final_plot <- ggplot(timeseries_csv, aes_string(x = x_axis_col, y = "value", color = "variable", group = "variable", linetype="variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, title), x = x_label, y = y_label)
  final_plot <- final_plot + scale_x_continuous(limits = c(lo_x_limit, hi_x_limit), breaks = as.numeric(unlist(breaks)))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = nice_column_names, values = color_array[1:length(nice_column_names)]) 
  final_plot <- final_plot + theme_bw() 
  final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  final_plot <- final_plot + theme(legend.position="bottom")
  final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
  
  ggsave(plot = final_plot, filename = out_img_path, width = 14) 

  # final_plot <- ggplot(timeseries_csv, aes_string(x = x_axis_col, y = "value", color = "variable", group = "variable")) 
  # final_plot <- final_plot + geom_line(aes(linetype=variable))
  # final_plot <- final_plot + labs(title = paste(title_prefix, title), x = x_label, y = y_label, color = "Legend")
  # final_plot <- final_plot + scale_x_continuous(limits = c(lo_x_limit, hi_x_limit), breaks = as.numeric(unlist(breaks)))
  # final_plot <- final_plot + scale_color_manual(labels = nice_column_names, values = color_array[1:length(nice_column_names)]) 
  # final_plot <- final_plot + theme_bw() 
  # final_plot <- final_plot + scale_linetype_manual(values = aes(variable)) 
  # final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
  # final_plot <- final_plot + theme(legend.position="bottom", legend.background = element_rect(colour = 'black', fill = 'white'))
  
  show(final_plot)
  ggsave(plot = final_plot, filename = out_img_path, width = 14) 
}

preprocess_column_names <- function(df) {
  column_names = colnames(df)
  column_names = gsub("_$", "", gsub("(\\.+)", "_", column_names, perl=TRUE), perl=TRUE)
  colnames(df) = column_names
  return(df)
}

plot_multi_line_timeseries_wrapper <- function(in_csv_paths, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
  if (missing(in_csv_paths)) {
    stop("Must specify the in CSV paths")
  }
  
  # Code which checks the parameters
  if (missing(x_axis_col)) {
    x_axis_col = "Node Count"
  }
  
  if (missing(y_axis_cols)) {
    stop("Must specify which the columns used for the Y axis")
  }
  
  # If not prefix is specified, default to ''. If it is specified, add a ' - ' after it.
  if (missing(title_prefix)) {
    title_prefix = ''
  } else {
    title_prefix = paste(title_prefix, " - ")
  }
  
  if (missing(title)) {
    stop("Title must be specified")
  }
  
  if (missing(x_label)) {
    x_label = "Node Count"
    warning("No X label specified. Will default to: 'Node Count'")
  }
  
  if (missing(y_label)) {
    stop("Must specify a Y label")
  }
  
  if (missing(nice_column_names)) {
    stop("Must provide nice column names")
  } else {
    if (length(nice_column_names) != length(y_axis_cols)) {
      stop("nice_column_names and y_axis_cols must have the same length")
    }
  }

  # Actual code
  i = 1
  l = vector("list", length(in_csv_paths))
  
  for (csv_name in in_csv_paths) {
    l[[i]] = preprocess_column_names(read.csv(csv_name))
    i = i + 1
  }

  plot_multi_line_timeseries(l, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path)
}

# Forwd_Seconds_per_Batch_sec_batch
# Forwd_Backwd_Seconds_per_Batch_sec_batch

csv_names = c(
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_c2.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_car_hp.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_cps_sp.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_dr_hp.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_ps_sp.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_dar_hp.csv',
  '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_dar_hp.csv'
)

# csv_names = c(
#   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_c2.csv",
#   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_car_hp.csv",
#   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_cps_hp.csv",
#   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_cps_sp.csv",
#   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_dr_hp.csv",
#   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_ps_sp.csv"
#   # "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_dar_hp.csv",
#   # "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_dar_hp.csv"
# )

# Forward and Backward Pass Sec / Batch - All
plot_multi_line_timeseries_wrapper(
  in_csv_paths = csv_names,
  x_axis_col = 'Node_Count',
  y_axis_cols = c(rep(c('Forwd_Backwd_Seconds_per_Batch_sec_batch'), length(csv_names) - 2), "RINGx1_Forwd_Backwd_Seconds_per_Batch_sec_batch", "PSCPU_Forwd_Backwd_Seconds_per_Batch_sec_batch"),
  title_prefix = '(Ethernet) Forward and Backward Pass',
  title = "Seconds per Batch Comparison",
  x_label = 'Node Count',
  y_label = 'Seconds / Batch',
  nice_column_names = c(
    "Caffe2",
    "Collective Allreduce",
    "Collocated Parameter Server",
    "(Soft Placement) Collocated Parameter Server",
    "Distributed Replicated",
    "Non-Collocated Parameter Server",
    "(Ringx1) Distributed Allreduce",
    "(PSCPU) Distributed Allreduce"
  ),
  out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_backward_all.png'
)

# Forward and Backward Pass Images / Sec - All
plot_multi_line_timeseries_wrapper(
  in_csv_paths = csv_names,
  x_axis_col = 'Node_Count',
  y_axis_cols = c(rep(c('Forwd_Backwd_Images_per_Second_img_sec'), length(csv_names) - 2), "RINGx1_Forwd_Backwd_Images_per_Second_img_sec", "PSCPU_Forwd_Backwd_Images_per_Second_img_sec"),
  title_prefix = '(Ethernet) Forward and Backward Pass',
  title = "Images per Second Comparison",
  x_label = 'Node Count',
  y_label = 'Images / Second',
  nice_column_names = c(
    "Caffe2",
    "Collective Allreduce",
    "Collocated Parameter Server",
    "(Soft Placement) Collocated Parameter Server",
    "Distributed Replicated",
    "Non-Collocated Parameter Server",
    "(Ringx1) Distributed Allreduce",
    "(PSCPU) Distributed Allreduce"
  ),
  out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_backward_all_images.png'
)

# Forward Pass Sec / Batch - All
# plot_multi_line_timeseries_wrapper(
#   in_csv_paths = csv_names,
#   x_axis_col = 'Node_Count',
#   y_axis_cols = rep(c('Forwd_Seconds_per_Batch_sec_batch'), length(csv_names)),
#   title_prefix = '(Ethernet) Forward Pass',
#   title = "Seconds per Batch Comparison",
#   x_label = 'Node Count',
#   y_label = 'Seconds / Batch',
#   nice_column_names = c(
#     "Caffe2",
#     "Collective Allreduce",
#     "Collocated Parameter Server",
#     "(Soft Placement) Collocated Parameter Server",
#     "Distributed Replicated",
#     "Non-Collocated Parameter Server"
#     # "(Ringx1) Distributed Allreduce",
#     # "(PSCPU) Distributed Allreduce"
#   ),
#   out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_all.png'
# )

# Forward Pass Images / Sec - All
# plot_multi_line_timeseries_wrapper(
#   in_csv_paths = csv_names,
#   x_axis_col = 'Node_Count',
#   y_axis_cols = rep(c('Forwd_Images_per_Second_img_sec'), length(csv_names)),
#   title_prefix = '(Ethernet) Forward Pass',
#   title = "Images per Second Comparison",
#   x_label = 'Node Count',
#   y_label = 'Images / Second',
#   nice_column_names = c(
#     "Caffe2",
#     "Collective Allreduce",
#     "Collocated Parameter Server",
#     "(Soft Placement) Collocated Parameter Server",
#     "Distributed Replicated",
#     "Non-Collocated Parameter Server"
#     # "(Ringx1) Distributed Allreduce",
#     # "(PSCPU) Distributed Allreduce"
#   ),
#   out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_all_images.png'
# )

###
### This is for the batch size scalability timeseries.
###


csv_names = c(
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n1_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n2_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n4_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n8_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n16_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n24_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n32_cps_hp.csv',
  '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n40_cps_hp.csv'
)

# csv_names = c(
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n1_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n2_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n4_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n8_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n16_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n24_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n32_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n40_cps_hp.csv'
# )

plot_multi_line_timeseries_wrapper(
  in_csv_paths = csv_names,
  x_axis_col = 'Batch_Size',
  y_axis_cols = rep(c('Forwd_Images_per_Second_img_sec'), length(csv_names)),
  title_prefix = '(Ethernet) Forward Pass',
  title = "Images per Second Comparison",
  x_label = 'Batch Size',
  y_label = 'Images / Second',
  nice_column_names = c(
    "N1",
    "N2",
    "N4",
    "N8",
    "N16",
    "N24",
    "N32",
    "N40"
  ),
  out_img_path = '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_forward_all_images_batch.png'
)









# library("base")
# library("dplyr")
# library("ggplot2")
# library("reshape2")
# 
# 
# gg_color_hue <- function(n) {
#   hues = seq(15, 375, length = n + 1)
#   hcl(h = hues, l = 65, c = 100)[1:n]
# }
# 
# color_array <- c("blue", "red", "green", "yellow", "purple", "orange", "cyan", "brown", "darkgreen", "darkorchid", "gold", "lightblue")
# 
# 
# plot_multi_line_timeseries <- function(timeseries_dfs, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
#   
#   # Filter the timeseries, select the relevant columns, and identify possible possitions for breaks in the plot
#   timeseries_csv = NULL
#   idx = 0
#   
#   # How many columns per file are there?
#   columns_per_file = length(y_axis_cols) / length(timeseries_dfs)
#   
#   for (df in timeseries_dfs) {
#     
#     temp_idx = idx * columns_per_file
#     
#     select_column_names = c(x_axis_col)
#     rename_column_names = c(x_axis_col)
#     
#     for (offset in 1:columns_per_file) {
#       select_column_names = c(select_column_names, y_axis_cols[temp_idx + offset])
#       rename_column_names = c(rename_column_names, paste(y_axis_cols[temp_idx + offset], idx, sep = "_"))
#     }
#     
#     temp = select(df, select_column_names)
#     colnames(temp) = rename_column_names
#     idx = idx + 1
#     
#     if (is.null(timeseries_csv)) {
#       timeseries_csv <- temp
#     } else {
#       timeseries_csv <- merge(timeseries_csv, temp, by = x_axis_col, all = TRUE)
#     }
#   }
#   
#   # Get the lower and upper bounds of the x axis, as well as the positions where the breaks occur
#   lo_x_limit = min(timeseries_csv[x_axis_col])
#   hi_x_limit = max(timeseries_csv[x_axis_col])
#   breaks = timeseries_csv[x_axis_col]
#   
#   # Melt the CSV to a plottable format
#   timeseries_csv = melt(timeseries_csv, id = x_axis_col)
#   
#   
#   # Plot formatting
#   final_plot <- ggplot(timeseries_csv, aes_string(x = x_axis_col, y = "value", color = "variable", group = "variable")) + geom_line()
#   final_plot <- final_plot + labs(title = paste(title_prefix, title), x = x_label, y = y_label)
#   final_plot <- final_plot + scale_x_continuous(limits = c(lo_x_limit, hi_x_limit), breaks = as.numeric(unlist(breaks)))
#   final_plot <- final_plot + labs(color = "Legend")
#   final_plot <- final_plot + scale_color_manual(labels = nice_column_names, values = color_array[1:length(nice_column_names)]) 
#   final_plot <- final_plot + theme_bw() 
#   final_plot <- final_plot + scale_fill_continuous(guide = guide_legend())
#   final_plot <- final_plot + theme(legend.position="bottom")
#   final_plot <- final_plot + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))
#   
#   ggsave(plot = final_plot, filename = out_img_path, width = 14) 
# }
# 
# preprocess_column_names <- function(df) {
#   column_names = colnames(df)
#   column_names = gsub("_$", "", gsub("(\\.+)", "_", column_names, perl=TRUE), perl=TRUE)
#   colnames(df) = column_names
#   return(df)
# }
# 
# plot_multi_line_timeseries_wrapper <- function(in_csv_paths, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
#   if (missing(in_csv_paths)) {
#     stop("Must specify the in CSV paths")
#   }
#   
#   # Code which checks the parameters
#   if (missing(x_axis_col)) {
#     x_axis_col = "Node Count"
#   }
#   
#   if (missing(y_axis_cols)) {
#     stop("Must specify which the columns used for the Y axis")
#   }
#   
#   # If not prefix is specified, default to ''. If it is specified, add a ' - ' after it.
#   if (missing(title_prefix)) {
#     title_prefix = ''
#   } else {
#     title_prefix = paste(title_prefix, " - ")
#   }
#   
#   if (missing(title)) {
#     stop("Title must be specified")
#   }
#   
#   if (missing(x_label)) {
#     x_label = "Node Count"
#     warning("No X label specified. Will default to: 'Node Count'")
#   }
#   
#   if (missing(y_label)) {
#     stop("Must specify a Y label")
#   }
#   
#   if (missing(nice_column_names)) {
#     stop("Must provide nice column names")
#   } else {
#     if (length(nice_column_names) != length(y_axis_cols)) {
#       stop("nice_column_names and y_axis_cols must have the same length")
#     }
#   }
#   
#   # Actual code
#   i = 1
#   l = vector("list", length(in_csv_paths))
#   
#   for (csv_name in in_csv_paths) {
#     l[[i]] = preprocess_column_names(read.csv(csv_name))
#     i = i + 1
#   }
#   
#   plot_multi_line_timeseries(l, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path)
# }
# 
# # Forwd_Seconds_per_Batch_sec_batch
# # Forwd_Backwd_Seconds_per_Batch_sec_batch
# 
# csv_names = c(
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_c2.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_car_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_cps_sp.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_dr_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_ps_sp.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_dar_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_dar_hp.csv'
# )
# 
# # csv_names = c(
# #   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_c2.csv",
# #   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_car_hp.csv",
# #   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_cps_hp.csv",
# #   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_cps_sp.csv",
# #   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_dr_hp.csv",
# #   "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_ps_sp.csv"
# #   # "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_dar_hp.csv",
# #   # "/data/files/university/msc_thesis/csv_results/collective_csv_plot/inf_dar_hp.csv"
# # )
# 
# # Forward and Backward Pass Sec / Batch - All
# plot_multi_line_timeseries_wrapper(
#   in_csv_paths = csv_names,
#   x_axis_col = 'Node_Count',
#   y_axis_cols = c(rep(c('Forwd_Backwd_Seconds_per_Batch_sec_batch'), length(csv_names) - 2), "RINGx1_Forwd_Backwd_Seconds_per_Batch_sec_batch", "PSCPU_Forwd_Backwd_Seconds_per_Batch_sec_batch"),
#   title_prefix = '(Ethernet) Forward and Backward Pass',
#   title = "Seconds per Batch Comparison",
#   x_label = 'Node Count',
#   y_label = 'Seconds / Batch',
#   nice_column_names = c(
#     "Caffe2",
#     "Collective Allreduce",
#     "Collocated Parameter Server",
#     "(Soft Placement) Collocated Parameter Server",
#     "Distributed Replicated",
#     "Non-Collocated Parameter Server",
#     "(Ringx1) Distributed Allreduce",
#     "(PSCPU) Distributed Allreduce"
#   ),
#   out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_backward_all.png'
# )
# 
# # Forward and Backward Pass Images / Sec - All
# plot_multi_line_timeseries_wrapper(
#   in_csv_paths = csv_names,
#   x_axis_col = 'Node_Count',
#   y_axis_cols = c(rep(c('Forwd_Backwd_Images_per_Second_img_sec'), length(csv_names) - 2), "RINGx1_Forwd_Backwd_Images_per_Second_img_sec", "PSCPU_Forwd_Backwd_Images_per_Second_img_sec"),
#   title_prefix = '(Ethernet) Forward and Backward Pass',
#   title = "Images per Second Comparison",
#   x_label = 'Node Count',
#   y_label = 'Images / Second',
#   nice_column_names = c(
#     "Caffe2",
#     "Collective Allreduce",
#     "Collocated Parameter Server",
#     "(Soft Placement) Collocated Parameter Server",
#     "Distributed Replicated",
#     "Non-Collocated Parameter Server",
#     "(Ringx1) Distributed Allreduce",
#     "(PSCPU) Distributed Allreduce"
#   ),
#   out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_backward_all_images.png'
# )
# 
# # Forward Pass Sec / Batch - All
# # plot_multi_line_timeseries_wrapper(
# #   in_csv_paths = csv_names,
# #   x_axis_col = 'Node_Count',
# #   y_axis_cols = rep(c('Forwd_Seconds_per_Batch_sec_batch'), length(csv_names)),
# #   title_prefix = '(Ethernet) Forward Pass',
# #   title = "Seconds per Batch Comparison",
# #   x_label = 'Node Count',
# #   y_label = 'Seconds / Batch',
# #   nice_column_names = c(
# #     "Caffe2",
# #     "Collective Allreduce",
# #     "Collocated Parameter Server",
# #     "(Soft Placement) Collocated Parameter Server",
# #     "Distributed Replicated",
# #     "Non-Collocated Parameter Server"
# #     # "(Ringx1) Distributed Allreduce",
# #     # "(PSCPU) Distributed Allreduce"
# #   ),
# #   out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_all.png'
# # )
# 
# # Forward Pass Images / Sec - All
# # plot_multi_line_timeseries_wrapper(
# #   in_csv_paths = csv_names,
# #   x_axis_col = 'Node_Count',
# #   y_axis_cols = rep(c('Forwd_Images_per_Second_img_sec'), length(csv_names)),
# #   title_prefix = '(Ethernet) Forward Pass',
# #   title = "Images per Second Comparison",
# #   x_label = 'Node Count',
# #   y_label = 'Images / Second',
# #   nice_column_names = c(
# #     "Caffe2",
# #     "Collective Allreduce",
# #     "Collocated Parameter Server",
# #     "(Soft Placement) Collocated Parameter Server",
# #     "Distributed Replicated",
# #     "Non-Collocated Parameter Server"
# #     # "(Ringx1) Distributed Allreduce",
# #     # "(PSCPU) Distributed Allreduce"
# #   ),
# #   out_img_path = '/data/files/university/msc_thesis/csv_results/collective_csv_plot/eth_forward_all_images.png'
# # )
# 
# ###
# ### This is for the batch size scalability timeseries.
# ###
# 
# 
# csv_names = c(
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n1_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n2_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n4_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n8_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n16_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n24_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n32_cps_hp.csv',
#   '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_n40_cps_hp.csv'
# )
# 
# # csv_names = c(
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n1_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n2_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n4_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n8_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n16_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n24_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n32_cps_hp.csv',
# #   '/data/files/university/msc_thesis/csv_results/batch_results/collective/inf_n40_cps_hp.csv'
# # )
# 
# plot_multi_line_timeseries_wrapper(
#   in_csv_paths = csv_names,
#   x_axis_col = 'Batch_Size',
#   y_axis_cols = rep(c('Forwd_Images_per_Second_img_sec'), length(csv_names)),
#   title_prefix = '(Ethernet) Forward Pass',
#   title = "Images per Second Comparison",
#   x_label = 'Batch Size',
#   y_label = 'Images / Second',
#   nice_column_names = c(
#     "N1",
#     "N2",
#     "N4",
#     "N8",
#     "N16",
#     "N24",
#     "N32",
#     "N40"
#   ),
#   out_img_path = '/data/files/university/msc_thesis/csv_results/batch_results/collective/eth_forward_all_images_batch.png'
# )