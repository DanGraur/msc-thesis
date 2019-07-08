library("base")
library("dplyr")
library("ggplot2")
library("reshape2")

# This function plots the CPU usage and Memory footprint given a dir path (which contains some CSVs)
plot_multi_line_timeseries <- function(timeseries_csv, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
  
  # Filter the timeseries, select the relevant columns, and identify possible possitions for breaks in the plot
  timeseries_csv = select(timeseries_csv, c(x_axis_col, y_axis_cols))
  
  # Get the lower and upper bounds of the x axis, as well as the positions where the breaks occur
  lo_x_limit = min(timeseries_csv[x_axis_col])
  hi_x_limit = max(timeseries_csv[x_axis_col])
  breaks = timeseries_csv[x_axis_col]
  
  # Melt the CSV to a plottable format
  timeseries_csv = melt(timeseries_csv, id = x_axis_col)
  
  final_plot <- ggplot(timeseries_csv, aes_string(x = x_axis_col, y = "value", color = "variable", group = "variable")) + geom_line()
  final_plot <- final_plot + labs(title = paste(title_prefix, title), x = x_label, y = y_label)
  final_plot <- final_plot + scale_x_continuous(limits = c(lo_x_limit, hi_x_limit), breaks = as.numeric(unlist(breaks)))
  final_plot <- final_plot + labs(color = "Legend")
  final_plot <- final_plot + scale_color_manual(labels = nice_column_names, values = c("blue", "red")) + theme_bw() + theme(legend.position=c(.26,.9)) + theme(legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid'))

  ggsave(plot = final_plot, filename = out_img_path, width = 14) 
}

preprocess_column_names <- function(df) {
  column_names = colnames(df)
  column_names = gsub("_$", "", gsub("(\\.+)", "_", column_names, perl=TRUE), perl=TRUE)
  colnames(df) = column_names
  return(df)
}

plot_multi_line_timeseries_wrapper <- function(in_csv_path, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
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
  df <- read.csv(in_csv_path)
  df <- preprocess_column_names(df)
  plot_multi_line_timeseries(df, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path)
}

plot_dir_single_timeseries <- function(dir_path, x_axis_col, y_axis_cols, title, x_label, y_label, nice_column_names) {
  csv_files <- list.files(path = dir_path, pattern = "csv")
  for (csv_file in csv_files) {
    out_img_path <- file.path(dir_path, sub(".csv", ".png", csv_file))
    
    title_prefix <- switch (basename(csv_file),
                            "eth_c2.csv" = "(Eth) Caffe2",
                            "eth_cps_sp.csv" = "(Eth) TensorFlow CPS Soft Placement",
                            "eth_ps_sp.csv" = "(Eth) TensorFlow PS Soft Placement",
                            "inf_cps_hp.csv" = "(Inf) TensorFlow CPS Hard Placement",
                            "inf_dr_hp.csv" = "(Inf) TensorFlow DR Hard Placement",
                            "eth_car_hp.csv" = "(Eth) TensorFlow CAR Hard Placement",
                            "eth_dar_hp.csv" = "(Eth) TensorFlow DAR Hard Placement",
                            "inf_c2.csv" = "(Inf) Caffe2",
                            "inf_cps_sp.csv" = "(Inf) TensorFlow CPS Soft Placement",
                            "inf_ps_sp.csv" = "(Inf) TensorFlow PS Soft Placement",
                            "eth_cps_hp.csv" = "(Eth) TensorFlow CPS Hard Placement",
                            "eth_dr_hp.csv" = "(Eth) TensorFlow DR Hard Placement",
                            "inf_car_hp.csv" = "(Inf) TensorFlow CAR Hard Placement",
                            "inf_dar_hp.csv" = "(Inf) TensorFlow DAR Hard Placement"
    )
    
    tryCatch(
      plot_multi_line_timeseries_wrapper(file.path(dir_path, csv_file), x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path),
      error=function(cond) {
        message(cond)
      }
    )
  }
}

# Plot all the singluar timeseries
# plot_dir_single_timeseries(
#     dir_path = '/data/files/university/msc_thesis/csv_results/csvs/',
#     x_axis_col = 'Node_Count',
#     y_axis_cols = c('Forwd_Images_per_Second_img_sec', 'Forwd_Backwd_Images_per_Second_img_sec'),
#     title = "Processed Images per Second",
#     x_label = 'Node Count',
#     y_label = 'Images / Second',
#     nice_column_names = c("Forward Pass", "Forward and Backward Pass")
#   )

# Plot a specific timeseries
plot_multi_line_timeseries_wrapper(
  in_csv_path = '/data/files/university/msc_thesis/csv_results/csvs/inf_dar_hp.csv',
  x_axis_col = 'Node_Count',
  y_axis_cols = c('RINGx1_Forwd_Backwd_Images_per_Second_img_sec', 'PSCPU_Forwd_Backwd_Images_per_Second_img_sec'),
  title_prefix = '(Inf) TensorFlow DAR',
  title = "Processed Images per Second",
  x_label = 'Node Count',
  y_label = 'Images / Second',
  nice_column_names = c("RINGx1", "PSCPU"),
  out_img_path = '/data/files/university/msc_thesis/csv_results/csvs/inf_dar_hp.png'
)
